package com.nofishing.controller;

import com.nofishing.dto.PaginatedResponse;
import com.nofishing.entity.DetectionHistory;
import com.nofishing.service.DetectionHistoryService;
import com.nofishing.util.ExcelGenerator;
import lombok.RequiredArgsConstructor;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

@RestController
@RequestMapping("/api/v1/history")
@RequiredArgsConstructor
public class DetectionHistoryController {

    private final DetectionHistoryService service;
    private final ExcelGenerator excelGenerator;

    @GetMapping
    public ResponseEntity<PaginatedResponse<DetectionHistory>> getHistory(
            @RequestParam(required = false) String keyword,
            @RequestParam(required = false) Boolean isPhishing,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        PaginatedResponse<DetectionHistory> history = service.getHistory(
                keyword, isPhishing, startTime, endTime, page, size);
        return ResponseEntity.ok(history);
    }

    @GetMapping("/{id}")
    public ResponseEntity<DetectionHistory> getById(@PathVariable Long id) {
        DetectionHistory history = service.getById(id);
        return ResponseEntity.ok(history);
    }

    @DeleteMapping("/{id}")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<Void> deleteById(@PathVariable Long id) {
        service.deleteById(id);
        return ResponseEntity.ok().build();
    }

    /**
     * Export detection history to Excel (Admin only)
     */
    @GetMapping("/export")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<byte[]> exportToExcel(
            @RequestParam(required = false) String keyword,
            @RequestParam(required = false) Boolean isPhishing,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime) throws IOException {

        List<DetectionHistory> data = service.getAllForExport(keyword, isPhishing, startTime, endTime);
        byte[] excel = excelGenerator.generateDetectionHistory(data);

        String filename = "detection_history_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")) + ".xlsx";

        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + filename + "\"")
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .body(excel);
    }
}
