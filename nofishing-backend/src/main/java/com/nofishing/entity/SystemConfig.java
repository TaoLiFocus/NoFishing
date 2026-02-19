package com.nofishing.entity;

import jakarta.persistence.*;
import lombok.Data;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

/**
 * System configuration entity for runtime configuration management
 */
@Entity
@Table(name = "sys_config")
@Data
@EntityListeners(AuditingEntityListener.class)
public class SystemConfig {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false, length = 50)
    private String configKey;

    @Column(length = 1000)
    private String configValue;

    @Column(length = 500)
    private String description;

    @Column(length = 50)
    private String category;

    @Column(length = 50)
    private String updatedBy;

    @CreatedDate
    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @LastModifiedDate
    @Column(nullable = false)
    private LocalDateTime updatedAt;

    /**
     * Configuration categories
     */
    public enum ConfigCategory {
        SYSTEM,
        ML,
        CACHE,
        SECURITY,
        FEATURE
    }
}
