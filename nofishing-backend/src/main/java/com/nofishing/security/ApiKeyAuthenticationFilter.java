package com.nofishing.security;

import com.nofishing.entity.ApiKey;
import com.nofishing.entity.User;
import com.nofishing.service.ApiKeyService;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.Collections;

/**
 * Filter for API Key authentication
 * Checks for X-API-Key header and validates it
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class ApiKeyAuthenticationFilter extends OncePerRequestFilter {

    private final ApiKeyService apiKeyService;

    private static final String API_KEY_HEADER = "X-API-Key";

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain) throws ServletException, IOException {

        String apiKeyValue = request.getHeader(API_KEY_HEADER);

        if (apiKeyValue != null && !apiKeyValue.isEmpty() && SecurityContextHolder.getContext().getAuthentication() == null) {
            try {
                ApiKey apiKey = apiKeyService.validateKey(apiKeyValue);

                if (apiKey != null) {
                    User user = apiKey.getUser();

                    // Create authentication with user's role
                    SimpleGrantedAuthority authority = new SimpleGrantedAuthority("ROLE_" + user.getRole().name());
                    UsernamePasswordAuthenticationToken authentication =
                            new UsernamePasswordAuthenticationToken(
                                    user.getUsername(),
                                    null,
                                    Collections.singletonList(authority)
                            );

                    authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                    SecurityContextHolder.getContext().setAuthentication(authentication);

                    log.debug("API key authenticated successfully for user: {}", user.getUsername());
                } else {
                    log.warn("Invalid or expired API key: {}", maskApiKey(apiKeyValue));
                }
            } catch (Exception e) {
                log.error("Error validating API key", e);
            }
        }

        filterChain.doFilter(request, response);
    }

    /**
     * Mask API key for logging (show only first 8 and last 4 characters)
     */
    private String maskApiKey(String apiKey) {
        if (apiKey == null || apiKey.length() < 12) {
            return "***";
        }
        return apiKey.substring(0, 8) + "..." + apiKey.substring(apiKey.length() - 4);
    }
}
