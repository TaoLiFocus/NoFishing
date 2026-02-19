package com.nofishing.controller;

import com.nofishing.service.DetectionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Health Check Controller
 *
 * @author NoFishing Team
 */
@Slf4j
@RestController
@RequestMapping("/api/v1")
@RequiredArgsConstructor
public class HealthController {

    private final DetectionService detectionService;

    /**
     * Health check endpoint
     *
     * GET /api/v1/health
     *
     * @return Health status
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> health = new HashMap<>();
        health.put("status", "UP");
        health.put("timestamp", LocalDateTime.now());
        health.put("service", "nofishing-backend");

        boolean mlServiceHealthy = detectionService.isMlServiceHealthy();
        health.put("mlService", mlServiceHealthy ? "UP" : "DOWN");

        return ResponseEntity.ok(health);
    }

    /**
     * Readiness probe
     *
     * GET /api/v1/ready
     *
     * @return Ready status
     */
    @GetMapping("/ready")
    public ResponseEntity<Map<String, String>> ready() {
        return ResponseEntity.ok(Map.of("status", "READY"));
    }
}
