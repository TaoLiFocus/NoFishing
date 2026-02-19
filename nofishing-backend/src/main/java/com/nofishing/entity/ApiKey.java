package com.nofishing.entity;

import jakarta.persistence.*;
import lombok.Data;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

/**
 * API Key entity for external API access
 */
@Entity
@Table(name = "api_key")
@Data
@EntityListeners(AuditingEntityListener.class)
public class ApiKey {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false, length = 64)
    private String keyValue;

    @Column(nullable = false, length = 100)
    private String name;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(length = 500)
    private String permissions;  // JSON: ["DETECT", "HISTORY_READ", "HISTORY_DELETE"]

    private LocalDateTime expiresAt;

    private LocalDateTime lastUsedAt;

    @Column(nullable = false)
    private Boolean isEnabled = true;

    @Column(length = 50)
    private String createdBy;

    @CreatedDate
    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;

    /**
     * Permission types
     */
    public enum Permission {
        DETECT,
        HISTORY_READ,
        HISTORY_DELETE,
        WHITELIST_READ,
        WHITELIST_WRITE,
        BLACKLIST_READ,
        BLACKLIST_WRITE,
        STATS_READ
    }

    /**
     * Check if key is expired
     */
    public boolean isExpired() {
        return expiresAt != null && LocalDateTime.now().isAfter(expiresAt);
    }

    /**
     * Check if key is valid (enabled and not expired)
     */
    public boolean isValid() {
        return isEnabled && !isExpired();
    }
}
