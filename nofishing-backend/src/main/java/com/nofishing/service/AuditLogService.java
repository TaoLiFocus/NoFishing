package com.nofishing.service;

import com.nofishing.dto.AuditLogDto;
import com.nofishing.dto.PaginatedResponse;
import com.nofishing.entity.AuditLog;
import com.nofishing.repository.AuditLogRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Service for managing audit logs
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AuditLogService {

    private final AuditLogRepository auditLogRepository;

    /**
     * Create audit log entry asynchronously
     */
    @Async
    public void createLogAsync(AuditLog auditLog) {
        try {
            auditLogRepository.save(auditLog);
        } catch (Exception e) {
            log.error("Failed to save audit log", e);
        }
    }

    /**
     * Search audit logs with filters
     */
    public PaginatedResponse<AuditLogDto> searchLogs(
            String operation,
            String module,
            String operatedBy,
            LocalDateTime startDate,
            LocalDateTime endDate,
            int page,
            int size) {

        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
        Page<AuditLog> result = auditLogRepository.searchLogs(
                operation, module, operatedBy, startDate, endDate, pageable
        );

        return PaginatedResponse.<AuditLogDto>builder()
                .content(result.getContent().stream().map(AuditLogDto::fromEntity).toList())
                .pageNumber(result.getNumber())
                .pageSize(result.getSize())
                .totalElements(result.getTotalElements())
                .totalPages(result.getTotalPages())
                .first(result.isFirst())
                .last(result.isLast())
                .build();
    }

    /**
     * Get recent audit logs
     */
    public List<AuditLogDto> getRecentLogs(int limit) {
        return auditLogRepository.findTop100ByOrderByCreatedAtDesc().stream()
                .limit(limit)
                .map(AuditLogDto::fromEntity)
                .toList();
    }

    /**
     * Get audit logs by operation type
     */
    public PaginatedResponse<AuditLogDto> getByOperation(String operation, int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
        Page<AuditLog> result = auditLogRepository.findByOperation(operation, pageable);

        return PaginatedResponse.<AuditLogDto>builder()
                .content(result.getContent().stream().map(AuditLogDto::fromEntity).toList())
                .pageNumber(result.getNumber())
                .pageSize(result.getSize())
                .totalElements(result.getTotalElements())
                .totalPages(result.getTotalPages())
                .first(result.isFirst())
                .last(result.isLast())
                .build();
    }

    /**
     * Get audit logs by module
     */
    public PaginatedResponse<AuditLogDto> getByModule(String module, int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
        Page<AuditLog> result = auditLogRepository.findByModule(module, pageable);

        return PaginatedResponse.<AuditLogDto>builder()
                .content(result.getContent().stream().map(AuditLogDto::fromEntity).toList())
                .pageNumber(result.getNumber())
                .pageSize(result.getSize())
                .totalElements(result.getTotalElements())
                .totalPages(result.getTotalPages())
                .first(result.isFirst())
                .last(result.isLast())
                .build();
    }

    /**
     * Get audit logs by operator
     */
    public PaginatedResponse<AuditLogDto> getByOperatedBy(String operatedBy, int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
        Page<AuditLog> result = auditLogRepository.findByOperatedBy(operatedBy, pageable);

        return PaginatedResponse.<AuditLogDto>builder()
                .content(result.getContent().stream().map(AuditLogDto::fromEntity).toList())
                .pageNumber(result.getNumber())
                .pageSize(result.getSize())
                .totalElements(result.getTotalElements())
                .totalPages(result.getTotalPages())
                .first(result.isFirst())
                .last(result.isLast())
                .build();
    }

    /**
     * Get audit logs by date range
     */
    public PaginatedResponse<AuditLogDto> getByDateRange(LocalDateTime startDate, LocalDateTime endDate, int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
        Page<AuditLog> result = auditLogRepository.findByCreatedAtBetween(startDate, endDate, pageable);

        return PaginatedResponse.<AuditLogDto>builder()
                .content(result.getContent().stream().map(AuditLogDto::fromEntity).toList())
                .pageNumber(result.getNumber())
                .pageSize(result.getSize())
                .totalElements(result.getTotalElements())
                .totalPages(result.getTotalPages())
                .first(result.isFirst())
                .last(result.isLast())
                .build();
    }

    /**
     * Get count of logs after a certain date
     */
    public long countAfterDate(LocalDateTime date) {
        return auditLogRepository.countByCreatedAtAfter(date);
    }
}
