package com.nofishing.repository;

import com.nofishing.entity.DetectionHistory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface DetectionHistoryRepository extends JpaRepository<DetectionHistory, Long> {
    Page<DetectionHistory> findByUrlContaining(String keyword, Pageable pageable);
    Page<DetectionHistory> findByIsPhishing(Boolean isPhishing, Pageable pageable);
    Page<DetectionHistory> findByDetectedAtBetween(LocalDateTime start, LocalDateTime end, Pageable pageable);
    Optional<DetectionHistory> findFirstByUrlOrderByDetectedAtDesc(String url);

    List<DetectionHistory> findByUrlContaining(String keyword);
    List<DetectionHistory> findByIsPhishing(Boolean isPhishing);
    List<DetectionHistory> findByDetectedAtBetween(LocalDateTime start, LocalDateTime end);

    long countByIsPhishingTrue();
    long countByDetectedAtBetween(LocalDateTime start, LocalDateTime end);
    long countByDetectedAtBetweenAndIsPhishingTrue(LocalDateTime start, LocalDateTime end);
    long countByRiskLevel(DetectionHistory.RiskLevel riskLevel);
}
