package com.nofishing.entity;

import jakarta.persistence.*;
import lombok.Data;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

/**
 * Audit log entity for tracking all critical operations
 */
@Entity
@Table(name = "audit_log")
@Data
@EntityListeners(AuditingEntityListener.class)
public class AuditLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(length = 50)
    private String operation;

    @Column(length = 50)
    private String module;

    @Column(length = 50)
    private String operatedBy;

    @Column(length = 50)
    private String targetType;

    private Long targetId;

    @Column(columnDefinition = "TEXT")
    private String targetValue;

    @Column(length = 50)
    private String ipAddress;

    @Column(columnDefinition = "VARCHAR(500)")
    private String userAgent;

    @Column(length = 50)
    private String status;

    @Column(columnDefinition = "TEXT")
    private String errorMessage;

    @CreatedDate
    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;

    /**
     * Operation types
     */
    public enum Operation {
        // User operations
        LOGIN, LOGOUT, CREATE_USER, UPDATE_USER, DELETE_USER, RESET_PASSWORD, CHANGE_PASSWORD,

        // Whitelist operations
        ADD_WHITELIST, DELETE_WHITELIST, UPDATE_WHITELIST,

        // Blacklist operations
        ADD_BLACKLIST, DELETE_BLACKLIST, UPDATE_BLACKLIST,

        // Detection history operations
        DELETE_HISTORY, BATCH_DELETE_HISTORY,

        // System config operations
        UPDATE_CONFIG, RESET_CONFIG,

        // API Key operations
        CREATE_API_KEY, REVOKE_API_KEY, UPDATE_API_KEY,

        // Batch operations
        BATCH_IMPORT, BATCH_DELETE
    }

    /**
     * Module types
     */
    public enum Module {
        AUTH, USER, WHITELIST, BLACKLIST, DETECTION, HISTORY, SYSTEM, API_KEY
    }

    /**
     * Status types
     */
    public enum Status {
        SUCCESS, FAILURE, PARTIAL
    }

    /**
     * Target types
     */
    public enum TargetType {
        USER, DOMAIN, RECORD, CONFIG, API_KEY, BATCH
    }
}
