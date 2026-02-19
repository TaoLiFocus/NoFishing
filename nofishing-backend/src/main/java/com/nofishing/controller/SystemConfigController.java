package com.nofishing.controller;

import com.nofishing.dto.SystemConfigDto;
import com.nofishing.dto.UpdateConfigRequest;
import com.nofishing.service.SystemConfigService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * Controller for system configuration management (Admin only)
 */
@Slf4j
@RestController
@RequestMapping("/api/v1/admin/system-config")
@RequiredArgsConstructor
@PreAuthorize("hasRole('ADMIN')")
public class SystemConfigController {

    private final SystemConfigService systemConfigService;

    /**
     * Get all configurations grouped by category
     */
    @GetMapping
    public ResponseEntity<Map<String, List<SystemConfigDto>>> getAllConfigs() {
        return ResponseEntity.ok(systemConfigService.getAllConfigsByCategory());
    }

    /**
     * Get configurations by category
     */
    @GetMapping("/category/{category}")
    public ResponseEntity<List<SystemConfigDto>> getConfigsByCategory(@PathVariable String category) {
        return ResponseEntity.ok(systemConfigService.getConfigsByCategory(category));
    }

    /**
     * Get single configuration value
     */
    @GetMapping("/{key}")
    public ResponseEntity<SystemConfigDto> getConfig(@PathVariable String key) {
        String value = systemConfigService.getConfig(key);
        SystemConfigDto dto = SystemConfigDto.builder()
                .configKey(key)
                .configValue(value)
                .build();
        return ResponseEntity.ok(dto);
    }

    /**
     * Update configuration value
     */
    @PutMapping("/{key}")
    public ResponseEntity<SystemConfigDto> updateConfig(
            @PathVariable String key,
            @Valid @RequestBody UpdateConfigRequest request,
            Authentication authentication) {
        String username = authentication.getName();
        com.nofishing.entity.SystemConfig updated = systemConfigService.updateConfig(
                key,
                request.getValue(),
                null,
                null,
                username
        );
        log.info("Config updated: {} by {}", key, username);
        return ResponseEntity.ok(SystemConfigDto.fromEntity(updated));
    }

    /**
     * Delete configuration
     */
    @DeleteMapping("/{key}")
    public ResponseEntity<Void> deleteConfig(@PathVariable String key) {
        systemConfigService.deleteConfig(key);
        return ResponseEntity.ok().build();
    }

    /**
     * Reset all configurations to default values
     */
    @PostMapping("/reset")
    public ResponseEntity<Void> resetToDefaults(Authentication authentication) {
        String username = authentication.getName();
        systemConfigService.resetToDefaults(username);
        log.info("All configs reset to defaults by {}", username);
        return ResponseEntity.ok().build();
    }

    /**
     * Initialize default configurations
     */
    @PostMapping("/initialize")
    public ResponseEntity<Void> initializeDefaults() {
        systemConfigService.initializeDefaults();
        return ResponseEntity.ok().build();
    }
}
