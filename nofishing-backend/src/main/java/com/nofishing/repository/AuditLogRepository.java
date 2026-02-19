package com.nofishing.repository;

import com.nofishing.entity.AuditLog;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface AuditLogRepository extends JpaRepository<AuditLog, Long> {

    Page<AuditLog> findByOperation(String operation, Pageable pageable);

    Page<AuditLog> findByModule(String module, Pageable pageable);

    Page<AuditLog> findByOperatedBy(String operatedBy, Pageable pageable);

    Page<AuditLog> findByCreatedAtBetween(LocalDateTime startDate, LocalDateTime endDate, Pageable pageable);

    @Query("SELECT a FROM AuditLog a WHERE " +
           "(:operation IS NULL OR a.operation = :operation) AND " +
           "(:module IS NULL OR a.module = :module) AND " +
           "(:operatedBy IS NULL OR a.operatedBy = :operatedBy) AND " +
           "(:startDate IS NULL OR a.createdAt >= :startDate) AND " +
           "(:endDate IS NULL OR a.createdAt <= :endDate)")
    Page<AuditLog> searchLogs(
            @Param("operation") String operation,
            @Param("module") String module,
            @Param("operatedBy") String operatedBy,
            @Param("startDate") LocalDateTime startDate,
            @Param("endDate") LocalDateTime endDate,
            Pageable pageable
    );

    List<AuditLog> findTop100ByOrderByCreatedAtDesc();

    long countByCreatedAtAfter(LocalDateTime date);
}
