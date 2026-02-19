package com.nofishing.entity;

import jakarta.persistence.*;
import lombok.Data;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

@Entity
@Table(name = "whitelist_entry")
@Data
@EntityListeners(AuditingEntityListener.class)
public class WhitelistEntry {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 500)
    private String pattern;

    @Column(length = 50)
    private String type;

    @Column(nullable = false)
    private Boolean enabled = true;

    @Column(length = 500)
    private String comment;

    @Column(length = 50)
    private String addedBy;

    @CreatedDate
    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column
    private LocalDateTime expiresAt;
}
