package com.nofishing.entity;

import jakarta.persistence.*;
import lombok.Data;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

@Entity
@Table(name = "blacklist_entry")
@Data
@EntityListeners(AuditingEntityListener.class)
public class BlacklistEntry {

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

    @Column(length = 50)
    private String threatType;

    @CreatedDate
    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column
    private LocalDateTime expiresAt;
}
