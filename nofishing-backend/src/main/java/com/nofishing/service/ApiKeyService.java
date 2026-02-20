package com.nofishing.service;

import com.nofishing.annotation.Audited;
import com.nofishing.entity.ApiKey;
import com.nofishing.entity.User;
import com.nofishing.repository.ApiKeyRepository;
import com.nofishing.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * Service for managing API keys
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ApiKeyService {

    private final ApiKeyRepository apiKeyRepository;
    private final UserRepository userRepository;

    /**
     * Create a new API key
     */
    @Transactional
    @Audited(operation = "CREATE_API_KEY", module = "API_KEY", targetType = "API_KEY")
    public ApiKey createKey(String name, Long userId, List<String> permissions, LocalDateTime expiresAt, String createdBy) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));

        // Check if user already has too many keys
        long keyCount = apiKeyRepository.findByUserId(userId).size();
        if (keyCount >= 10) {
            throw new RuntimeException("User has reached maximum API key limit (10)");
        }

        ApiKey apiKey = new ApiKey();
        apiKey.setKeyValue(generateKeyValue());
        apiKey.setName(name);
        apiKey.setUser(user);
        apiKey.setPermissions(String.join(",", permissions));
        apiKey.setExpiresAt(expiresAt);
        apiKey.setCreatedBy(createdBy);

        ApiKey saved = apiKeyRepository.save(apiKey);
        log.info("Created API key '{}' for user {} by {}", name, userId, createdBy);
        return saved;
    }

    /**
     * Revoke (delete) an API key
     */
    @Transactional
    @Audited(operation = "REVOKE_API_KEY", module = "API_KEY", targetType = "API_KEY")
    public void revokeKey(Long keyId) {
        apiKeyRepository.deleteById(keyId);
        log.info("Revoked API key with id: {}", keyId);
    }

    /**
     * Disable an API key (soft delete)
     */
    @Transactional
    @Audited(operation = "UPDATE_API_KEY", module = "API_KEY", targetType = "API_KEY")
    public void disableKey(Long keyId) {
        ApiKey apiKey = apiKeyRepository.findById(keyId)
                .orElseThrow(() -> new RuntimeException("API key not found"));
        apiKey.setIsEnabled(false);
        apiKeyRepository.save(apiKey);
        log.info("Disabled API key with id: {}", keyId);
    }

    /**
     * Get all keys for a user
     */
    public List<ApiKey> getUserKeys(Long userId) {
        return apiKeyRepository.findByUserIdWithUser(userId);
    }

    /**
     * Get all keys for a user (only enabled)
     */
    public List<ApiKey> getEnabledUserKeys(Long userId) {
        return apiKeyRepository.findByUserIdAndIsEnabledTrue(userId);
    }

    /**
     * Validate API key and check permission
     */
    @Transactional(readOnly = true)
    public ApiKey validateKey(String keyValue, String requiredPermission) {
        // Use eager fetch to avoid LazyInitializationException when accessing user.role
        ApiKey apiKey = apiKeyRepository.findByKeyValueWithUser(keyValue).orElse(null);

        if (apiKey == null) {
            return null;
        }

        if (!apiKey.isValid()) {
            return null;
        }

        if (requiredPermission != null && !hasPermission(apiKey, requiredPermission)) {
            return null;
        }

        // Update last used time asynchronously
        updateLastUsedTimeAsync(apiKey.getId());

        return apiKey;
    }

    /**
     * Validate API key (without permission check)
     */
    public ApiKey validateKey(String keyValue) {
        return validateKey(keyValue, null);
    }

    /**
     * Check if API key has a specific permission
     */
    private boolean hasPermission(ApiKey apiKey, String permission) {
        if (apiKey.getPermissions() == null || apiKey.getPermissions().isEmpty()) {
            return false;
        }

        List<String> permissions = List.of(apiKey.getPermissions().split(","));
        return permissions.contains(permission);
    }

    /**
     * Update last used time asynchronously
     */
    @Async
    protected void updateLastUsedTime(Long keyId) {
        try {
            apiKeyRepository.findById(keyId).ifPresent(key -> {
                key.setLastUsedAt(LocalDateTime.now());
                apiKeyRepository.save(key);
            });
        } catch (Exception e) {
            log.error("Failed to update last used time for API key: {}", keyId, e);
        }
    }

    /**
     * Update last used time asynchronously (method alias)
     */
    @Async
    public void updateLastUsedTimeAsync(Long keyId) {
        updateLastUsedTime(keyId);
    }

    /**
     * Generate a unique API key value
     */
    private String generateKeyValue() {
        String keyValue;
        do {
            keyValue = "nf_" + UUID.randomUUID().toString().replace("-", "");
        } while (apiKeyRepository.existsByKeyValue(keyValue));
        return keyValue;
    }

    /**
     * Clean up expired keys
     */
    @Transactional
    public void cleanupExpiredKeys() {
        List<ApiKey> expiredKeys = apiKeyRepository.findByExpiresAtBefore(LocalDateTime.now());
        for (ApiKey key : expiredKeys) {
            key.setIsEnabled(false);
            apiKeyRepository.save(key);
        }
        log.info("Disabled {} expired API keys", expiredKeys.size());
    }

    /**
     * Get all API keys (admin only)
     */
    public List<ApiKey> getAllKeys() {
        return apiKeyRepository.findAllWithUser();
    }

    /**
     * Get API key by ID
     */
    public ApiKey getKeyById(Long keyId) {
        return apiKeyRepository.findById(keyId)
                .orElseThrow(() -> new RuntimeException("API key not found"));
    }
}
