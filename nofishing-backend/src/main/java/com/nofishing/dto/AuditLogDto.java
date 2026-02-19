package com.nofishing.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * DTO for AuditLog
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AuditLogDto {

    private Long id;
    private String operation;
    private String module;
    private String operatedBy;
    private String targetType;
    private Long targetId;
    private String targetValue;
    private String ipAddress;
    private String userAgent;
    private String status;
    private String errorMessage;
    private LocalDateTime createdAt;

    public static AuditLogDto fromEntity(com.nofishing.entity.AuditLog entity) {
        return AuditLogDto.builder()
                .id(entity.getId())
                .operation(entity.getOperation())
                .module(entity.getModule())
                .operatedBy(entity.getOperatedBy())
                .targetType(entity.getTargetType())
                .targetId(entity.getTargetId())
                .targetValue(entity.getTargetValue())
                .ipAddress(entity.getIpAddress())
                .userAgent(entity.getUserAgent())
                .status(entity.getStatus())
                .errorMessage(entity.getErrorMessage())
                .createdAt(entity.getCreatedAt())
                .build();
    }
}
