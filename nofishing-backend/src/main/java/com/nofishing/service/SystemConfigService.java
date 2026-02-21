package com.nofishing.service;

import com.nofishing.annotation.Audited;
import com.nofishing.dto.SystemConfigDto;
import com.nofishing.entity.SystemConfig;
import com.nofishing.repository.SystemConfigRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Service for managing system configurations with caching support
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class SystemConfigService {

    private final SystemConfigRepository repository;

    // Default configuration values
    private static final Map<String, String> DEFAULT_CONFIGS = new HashMap<>();

    static {
        DEFAULT_CONFIGS.put("detection.threshold", "0.5");
        DEFAULT_CONFIGS.put("cache.ttl", "3600");
        DEFAULT_CONFIGS.put("cache.max-size", "10000");
        DEFAULT_CONFIGS.put("ml.service.timeout", "3000");
        DEFAULT_CONFIGS.put("ml.service.connect-timeout", "1000");
        DEFAULT_CONFIGS.put("ml.service.read-timeout", "2000");
        DEFAULT_CONFIGS.put("registration.enabled", "true");
        DEFAULT_CONFIGS.put("maintenance.mode", "false");
    }

    /**
     * Get configuration value by key with caching
     */
    @Cacheable(value = "systemConfigs", key = "#configKey")
    public String getConfig(String configKey) {
        return getConfig(configKey, DEFAULT_CONFIGS.get(configKey));
    }

    /**
     * Get configuration value by key with default fallback
     */
    public String getConfig(String configKey, String defaultValue) {
        Optional<SystemConfig> config = repository.findByConfigKey(configKey);
        return config.map(SystemConfig::getConfigValue).orElse(defaultValue);
    }

    /**
     * Get configuration as Integer
     */
    public Integer getIntConfig(String configKey, Integer defaultValue) {
        String value = getConfig(configKey);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            log.warn("Invalid integer value for config key: {}", configKey);
            return defaultValue;
        }
    }

    /**
     * Get configuration as Double
     */
    public Double getDoubleConfig(String configKey, Double defaultValue) {
        String value = getConfig(configKey);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            log.warn("Invalid double value for config key: {}", configKey);
            return defaultValue;
        }
    }

    /**
     * Get configuration as Boolean
     */
    public Boolean getBooleanConfig(String configKey, Boolean defaultValue) {
        String value = getConfig(configKey);
        if (value == null) {
            return defaultValue;
        }
        return Boolean.parseBoolean(value);
    }

    /**
     * Update or create configuration
     */
    @Transactional
    @Audited(operation = "UPDATE_CONFIG", module = "SYSTEM", targetType = "CONFIG")
    @CacheEvict(value = "systemConfigs", key = "#configKey")
    public SystemConfig updateConfig(String configKey, String configValue, String description, String category, String updatedBy) {
        Optional<SystemConfig> existing = repository.findByConfigKey(configKey);

        if (existing.isPresent()) {
            SystemConfig config = existing.get();
            config.setConfigValue(configValue);

            // Only update category if provided
            if (category != null) {
                config.setCategory(category);
            }

            // Only update description if provided
            if (description != null) {
                config.setDescription(description);
            }

            // Ensure category and description are set if missing
            if (config.getCategory() == null || config.getCategory().isEmpty()) {
                config.setCategory(getCategoryForKey(configKey));
            }
            if (config.getDescription() == null || config.getDescription().isEmpty()) {
                config.setDescription(getDescriptionForKey(configKey));
            }

            config.setUpdatedBy(updatedBy);
            log.info("Updated config: {} = {} by {}", configKey, configValue, updatedBy);
            return repository.save(config);
        } else {
            SystemConfig newConfig = new SystemConfig();
            newConfig.setConfigKey(configKey);
            newConfig.setConfigValue(configValue);

            // Set default description if not provided
            if (description == null || description.isEmpty()) {
                newConfig.setDescription(getDescriptionForKey(configKey));
            } else {
                newConfig.setDescription(description);
            }

            // Set default category if not provided
            if (category == null || category.isEmpty()) {
                newConfig.setCategory(getCategoryForKey(configKey));
            } else {
                newConfig.setCategory(category);
            }

            newConfig.setUpdatedBy(updatedBy);
            log.info("Created config: {} = {} by {}", configKey, configValue, updatedBy);
            return repository.save(newConfig);
        }
    }

    /**
     * Get all configurations grouped by category
     */
    public Map<String, List<SystemConfigDto>> getAllConfigsByCategory() {
        List<SystemConfig> allConfigs = repository.findAll();
        Map<String, List<SystemConfigDto>> result = new HashMap<>();

        for (SystemConfig config : allConfigs) {
            String category = config.getCategory() != null ? config.getCategory() : "OTHER";
            result.computeIfAbsent(category, k -> new java.util.ArrayList<>())
                    .add(SystemConfigDto.fromEntity(config));
        }

        return result;
    }

    /**
     * Get configurations by category
     */
    public List<SystemConfigDto> getConfigsByCategory(String category) {
        return repository.findByCategoryOrderByConfigKey(category).stream()
                .map(SystemConfigDto::fromEntity)
                .toList();
    }

    /**
     * Get all configurations
     */
    public List<SystemConfigDto> getAllConfigs() {
        return repository.findAll().stream()
                .map(SystemConfigDto::fromEntity)
                .toList();
    }

    /**
     * Delete configuration
     */
    @Transactional
    @CacheEvict(value = "systemConfigs", key = "#configKey")
    public void deleteConfig(String configKey) {
        repository.findByConfigKey(configKey).ifPresentOrElse(
                config -> {
                    repository.delete(config);
                    log.info("Deleted config: {}", configKey);
                },
                () -> log.warn("Config not found for deletion: {}", configKey)
        );
    }

    /**
     * Reset all configurations to defaults
     */
    @Transactional
    @Audited(operation = "RESET_CONFIG", module = "SYSTEM", targetType = "BATCH")
    @CacheEvict(value = "systemConfigs", allEntries = true)
    public void resetToDefaults(String updatedBy) {
        for (Map.Entry<String, String> entry : DEFAULT_CONFIGS.entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue();
            String category = getCategoryForKey(key);
            String description = getDescriptionForKey(key);

            Optional<SystemConfig> existing = repository.findByConfigKey(key);
            if (existing.isPresent()) {
                SystemConfig config = existing.get();
                config.setConfigValue(value);
                config.setUpdatedBy(updatedBy);
                repository.save(config);
            } else {
                SystemConfig newConfig = new SystemConfig();
                newConfig.setConfigKey(key);
                newConfig.setConfigValue(value);
                newConfig.setCategory(category);
                newConfig.setDescription(description);
                newConfig.setUpdatedBy(updatedBy);
                repository.save(newConfig);
            }
        }
        log.info("Reset all configs to defaults by {}", updatedBy);
    }

    /**
     * Initialize default configurations if they don't exist
     */
    @Transactional
    public void initializeDefaults() {
        for (Map.Entry<String, String> entry : DEFAULT_CONFIGS.entrySet()) {
            String key = entry.getKey();
            if (!repository.existsByConfigKey(key)) {
                SystemConfig config = new SystemConfig();
                config.setConfigKey(key);
                config.setConfigValue(entry.getValue());
                config.setCategory(getCategoryForKey(key));
                config.setDescription(getDescriptionForKey(key));
                config.setUpdatedBy("SYSTEM");
                repository.save(config);
            }
        }
    }

    private String getCategoryForKey(String key) {
        if (key.startsWith("detection.")) return "ML";
        if (key.startsWith("cache.")) return "CACHE";
        if (key.startsWith("ml.")) return "ML";
        if (key.startsWith("registration.") || key.startsWith("maintenance.")) return "SECURITY";
        return "SYSTEM";
    }

    private String getDescriptionForKey(String key) {
        return switch (key) {
            case "detection.threshold" -> "钓鱼检测阈值 (0.0-1.0)";
            case "cache.ttl" -> "缓存过期时间(秒)";
            case "cache.max-size" -> "缓存最大条目数";
            case "ml.service.timeout" -> "ML服务超时(毫秒)";
            case "ml.service.connect-timeout" -> "ML服务连接超时(毫秒)";
            case "ml.service.read-timeout" -> "ML服务读取超时(毫秒)";
            case "registration.enabled" -> "是否开放用户注册";
            case "maintenance.mode" -> "维护模式（禁用检测）";
            default -> "";
        };
    }
}
