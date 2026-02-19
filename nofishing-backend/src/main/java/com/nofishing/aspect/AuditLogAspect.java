package com.nofishing.aspect;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nofishing.annotation.Audited;
import com.nofishing.entity.AuditLog;
import com.nofishing.repository.AuditLogRepository;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.reflect.MethodSignature;
import org.springframework.scheduling.annotation.Async;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.lang.reflect.Method;
import java.time.LocalDateTime;
import java.util.Arrays;

/**
 * AOP Aspect for processing @Audited annotation
 * Logs operations asynchronously to avoid performance impact
 */
@Slf4j
@Aspect
@Component
@RequiredArgsConstructor
public class AuditLogAspect {

    private final AuditLogRepository auditLogRepository;
    private final ObjectMapper objectMapper;

    /**
     * Around advice for methods annotated with @Audited
     */
    @Around("@annotation(com.nofishing.annotation.Audited)")
    public Object logAudit(ProceedingJoinPoint joinPoint) throws Throwable {
        MethodSignature signature = (MethodSignature) joinPoint.getSignature();
        // Get the actual method from the target class to find the annotation
        Method method = getMethodFromTarget(joinPoint.getTarget().getClass(), signature.getMethod());
        Audited audited = method.getAnnotation(Audited.class);

        log.info("Audit AOP triggered for: {}.{}", method.getDeclaringClass().getSimpleName(), method.getName());

        if (audited == null) {
            log.error("@Audited annotation not found on method: {}", method);
            return joinPoint.proceed();
        }

        AuditLog auditLog = new AuditLog();
        auditLog.setOperation(audited.operation());
        auditLog.setModule(audited.module());
        auditLog.setTargetType(audited.targetType());

        // Get current user
        String username = getCurrentUsername();
        auditLog.setOperatedBy(username);

        // Get request info
        HttpServletRequest request = getCurrentRequest();
        if (request != null) {
            auditLog.setIpAddress(getClientIpAddress(request));
            auditLog.setUserAgent(request.getHeader("User-Agent"));
        }

        // Log parameters if enabled
        if (audited.logParams()) {
            try {
                Object[] args = joinPoint.getArgs();
                String paramsJson = objectMapper.writeValueAsString(args);
                auditLog.setTargetValue(truncate(paramsJson, 1000));
            } catch (Exception e) {
                log.warn("Failed to serialize method parameters", e);
            }
        }

        Object result = null;
        try {
            result = joinPoint.proceed();
            auditLog.setStatus(AuditLog.Status.SUCCESS.name());
            return result;
        } catch (Exception e) {
            auditLog.setStatus(AuditLog.Status.FAILURE.name());
            auditLog.setErrorMessage(truncate(e.getMessage(), 500));
            throw e;
        } finally {
            auditLog.setCreatedAt(LocalDateTime.now());
            // Save asynchronously to avoid blocking
            saveAuditLogAsync(auditLog);
        }
    }

    /**
     * Get current username from security context
     */
    private String getCurrentUsername() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication != null && authentication.isAuthenticated()) {
            return authentication.getName();
        }
        return "ANONYMOUS";
    }

    /**
     * Get current HTTP request
     */
    private HttpServletRequest getCurrentRequest() {
        ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        return attributes != null ? attributes.getRequest() : null;
    }

    /**
     * Get client IP address from request
     */
    private String getClientIpAddress(HttpServletRequest request) {
        String ip = request.getHeader("X-Forwarded-For");
        if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getHeader("X-Real-IP");
        }
        if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getHeader("Proxy-Client-IP");
        }
        if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getHeader("WL-Proxy-Client-IP");
        }
        if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getRemoteAddr();
        }
        // Handle multiple IPs in X-Forwarded-For
        if (ip != null && ip.contains(",")) {
            ip = ip.split(",")[0].trim();
        }
        return ip;
    }

    /**
     * Truncate string to max length
     */
    private String truncate(String str, int maxLength) {
        if (str == null) {
            return null;
        }
        if (str.length() <= maxLength) {
            return str;
        }
        return str.substring(0, maxLength) + "...";
    }

    /**
     * Get the actual method from target class
     * Handles cases where the method might be from an interface
     */
    private Method getMethodFromTarget(Class<?> targetClass, Method method) throws NoSuchMethodException {
        try {
            // Try to get the declared method from target class
            return targetClass.getDeclaredMethod(method.getName(), method.getParameterTypes());
        } catch (NoSuchMethodException e) {
            // If not found in target class, search in superclass hierarchy
            Class<?> currentClass = targetClass;
            while (currentClass != null && currentClass != Object.class) {
                try {
                    return currentClass.getDeclaredMethod(method.getName(), method.getParameterTypes());
                } catch (NoSuchMethodException ex) {
                    currentClass = currentClass.getSuperclass();
                }
            }
            // Fallback to original method
            return method;
        }
    }

    /**
     * Save audit log asynchronously
     */
    @Async
    protected void saveAuditLogAsync(AuditLog auditLog) {
        try {
            auditLogRepository.save(auditLog);
            log.debug("Audit log saved: {} {} by {}", auditLog.getOperation(), auditLog.getModule(), auditLog.getOperatedBy());
        } catch (Exception e) {
            log.error("Failed to save audit log", e);
        }
    }
}
