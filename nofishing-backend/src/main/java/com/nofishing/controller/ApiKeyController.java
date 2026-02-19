package com.nofishing.controller;

import com.nofishing.dto.ApiKeyDto;
import com.nofishing.dto.CreateApiKeyRequest;
import com.nofishing.entity.ApiKey;
import com.nofishing.service.ApiKeyService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Controller for API key management (Admin only)
 */
@Slf4j
@RestController
@RequestMapping("/api/v1/admin/api-keys")
@RequiredArgsConstructor
@PreAuthorize("hasRole('ADMIN')")
@Transactional(readOnly = true)
public class ApiKeyController {

    private final ApiKeyService apiKeyService;

    /**
     * Create a new API key
     */
    @Transactional
    @PostMapping
    public ResponseEntity<ApiKeyDto> createKey(
            @Valid @RequestBody CreateApiKeyRequest request,
            Authentication authentication) {

        String createdBy = authentication.getName();

        // Set default expiration to 1 year if not provided
        LocalDateTime expiresAt = request.getExpiresAt();
        if (expiresAt == null) {
            expiresAt = LocalDateTime.now().plusYears(1);
        }

        ApiKey apiKey = apiKeyService.createKey(
                request.getName(),
                request.getUserId(),
                request.getPermissions(),
                expiresAt,
                createdBy
        );

        log.info("API key created by {} for user {}", createdBy, request.getUserId());
        return ResponseEntity.ok(ApiKeyDto.fromEntity(apiKey));
    }

    /**
     * Get all API keys for a user
     */
    @GetMapping("/user/{userId}")
    public ResponseEntity<List<ApiKeyDto>> getUserKeys(@PathVariable Long userId) {
        List<ApiKey> keys = apiKeyService.getUserKeys(userId);
        return ResponseEntity.ok(keys.stream().map(ApiKeyDto::fromEntity).toList());
    }

    /**
     * Get all API keys (admin only)
     */
    @GetMapping
    public ResponseEntity<List<ApiKeyDto>> getAllKeys() {
        List<ApiKey> keys = apiKeyService.getAllKeys();
        return ResponseEntity.ok(keys.stream().map(ApiKeyDto::fromEntity).toList());
    }

    /**
     * Get API key by ID
     */
    @GetMapping("/{id}")
    public ResponseEntity<ApiKeyDto> getKeyById(@PathVariable Long id) {
        ApiKey key = apiKeyService.getKeyById(id);
        return ResponseEntity.ok(ApiKeyDto.fromEntity(key));
    }

    /**
     * Revoke (delete) an API key
     */
    @Transactional
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> revokeKey(@PathVariable Long id) {
        apiKeyService.revokeKey(id);
        log.info("API key {} revoked", id);
        return ResponseEntity.ok().build();
    }

    /**
     * Disable an API key (soft delete)
     */
    @Transactional
    @PostMapping("/{id}/disable")
    public ResponseEntity<Void> disableKey(@PathVariable Long id) {
        apiKeyService.disableKey(id);
        log.info("API key {} disabled", id);
        return ResponseEntity.ok().build();
    }

    /**
     * Clean up expired keys
     */
    @Transactional
    @PostMapping("/cleanup")
    public ResponseEntity<Void> cleanupExpiredKeys() {
        apiKeyService.cleanupExpiredKeys();
        return ResponseEntity.ok().build();
    }
}
