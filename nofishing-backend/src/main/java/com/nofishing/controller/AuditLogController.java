package com.nofishing.controller;

import com.nofishing.dto.AuditLogDto;
import com.nofishing.dto.PaginatedResponse;
import com.nofishing.service.AuditLogService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Controller for audit log management (Admin only)
 */
@Slf4j
@RestController
@RequestMapping("/api/v1/admin/audit-logs")
@RequiredArgsConstructor
@PreAuthorize("hasRole('ADMIN')")
public class AuditLogController {

    private final AuditLogService auditLogService;

    /**
     * Search audit logs with filters
     */
    @GetMapping
    public ResponseEntity<PaginatedResponse<AuditLogDto>> searchLogs(
            @RequestParam(required = false) String operation,
            @RequestParam(required = false) String module,
            @RequestParam(required = false) String operatedBy,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {

        return ResponseEntity.ok(auditLogService.searchLogs(
                operation, module, operatedBy, startDate, endDate, page, size
        ));
    }

    /**
     * Get recent audit logs
     */
    @GetMapping("/recent")
    public ResponseEntity<List<AuditLogDto>> getRecentLogs(
            @RequestParam(defaultValue = "50") int limit) {
        return ResponseEntity.ok(auditLogService.getRecentLogs(limit));
    }

    /**
     * Get audit logs by operation type
     */
    @GetMapping("/operation/{operation}")
    public ResponseEntity<PaginatedResponse<AuditLogDto>> getByOperation(
            @PathVariable String operation,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {

        return ResponseEntity.ok(auditLogService.getByOperation(operation, page, size));
    }

    /**
     * Get audit logs by module
     */
    @GetMapping("/module/{module}")
    public ResponseEntity<PaginatedResponse<AuditLogDto>> getByModule(
            @PathVariable String module,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {

        return ResponseEntity.ok(auditLogService.getByModule(module, page, size));
    }

    /**
     * Get audit logs by operator
     */
    @GetMapping("/operator/{operatedBy}")
    public ResponseEntity<PaginatedResponse<AuditLogDto>> getByOperatedBy(
            @PathVariable String operatedBy,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {

        return ResponseEntity.ok(auditLogService.getByOperatedBy(operatedBy, page, size));
    }

    /**
     * Get audit logs by date range
     */
    @GetMapping("/date-range")
    public ResponseEntity<PaginatedResponse<AuditLogDto>> getByDateRange(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {

        return ResponseEntity.ok(auditLogService.getByDateRange(startDate, endDate, page, size));
    }
}
