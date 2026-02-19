package com.nofishing.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * DTO for ApiKey
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ApiKeyDto {

    private Long id;
    private String keyValue;
    private String name;
    private Long userId;
    private String username;
    private List<String> permissions;
    private LocalDateTime expiresAt;
    private LocalDateTime lastUsedAt;
    private Boolean isEnabled;
    private String createdBy;
    private LocalDateTime createdAt;

    public static ApiKeyDto fromEntity(com.nofishing.entity.ApiKey entity) {
        // Handle permissions - convert comma-separated string to list
        List<String> permissionsList = new ArrayList<>();
        if (entity.getPermissions() != null && !entity.getPermissions().isEmpty()) {
            String[] parts = entity.getPermissions().split(",");
            for (String part : parts) {
                String trimmed = part.trim();
                if (!trimmed.isEmpty()) {
                    permissionsList.add(trimmed);
                }
            }
        }

        // Get user info safely
        Long userId = null;
        String username = null;
        if (entity.getUser() != null) {
            userId = entity.getUser().getId();
            username = entity.getUser().getUsername();
        }

        return ApiKeyDto.builder()
                .id(entity.getId())
                .keyValue(entity.getKeyValue())
                .name(entity.getName())
                .userId(userId)
                .username(username)
                .permissions(permissionsList.isEmpty() ? Collections.emptyList() : permissionsList)
                .expiresAt(entity.getExpiresAt())
                .lastUsedAt(entity.getLastUsedAt())
                .isEnabled(entity.getIsEnabled())
                .createdBy(entity.getCreatedBy())
                .createdAt(entity.getCreatedAt())
                .build();
    }
}
