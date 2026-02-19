package com.nofishing.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * DTO for SystemConfig
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SystemConfigDto {

    private Long id;
    private String configKey;
    private String configValue;
    private String description;
    private String category;
    private String updatedBy;
    private String createdAt;
    private String updatedAt;

    public static SystemConfigDto fromEntity(com.nofishing.entity.SystemConfig entity) {
        return SystemConfigDto.builder()
                .id(entity.getId())
                .configKey(entity.getConfigKey())
                .configValue(entity.getConfigValue())
                .description(entity.getDescription())
                .category(entity.getCategory())
                .updatedBy(entity.getUpdatedBy())
                .createdAt(entity.getCreatedAt() != null ? entity.getCreatedAt().toString() : null)
                .updatedAt(entity.getUpdatedAt() != null ? entity.getUpdatedAt().toString() : null)
                .build();
    }
}
