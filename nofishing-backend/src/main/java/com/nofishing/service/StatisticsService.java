package com.nofishing.service;

import com.nofishing.entity.DetectionHistory;
import com.nofishing.repository.DetectionHistoryRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.*;

@Service
@RequiredArgsConstructor
public class StatisticsService {

    private final DetectionHistoryRepository historyRepository;

    public Map<String, Object> getSummary() {
        long totalDetections = historyRepository.count();
        long phishingCount = historyRepository.countByIsPhishingTrue();
        long safeCount = totalDetections - phishingCount;
        double phishingRate = totalDetections > 0 ? (double) phishingCount / totalDetections * 100 : 0;

        Map<String, Object> summary = new HashMap<>();
        summary.put("totalDetections", totalDetections);
        summary.put("phishingCount", phishingCount);
        summary.put("safeCount", safeCount);
        summary.put("phishingRate", Math.round(phishingRate * 100) / 100.0);

        return summary;
    }

    public List<Map<String, Object>> getTrendData(LocalDateTime startTime, LocalDateTime endTime) {
        List<Map<String, Object>> trendData = new ArrayList<>();

        for (int i = 0; i < 7; i++) {
            LocalDateTime dayStart = startTime.plusDays(i);
            LocalDateTime dayEnd = dayStart.plusDays(1);

            long total = historyRepository.countByDetectedAtBetween(dayStart, dayEnd);
            long phishing = historyRepository.countByDetectedAtBetweenAndIsPhishingTrue(dayStart, dayEnd);

            Map<String, Object> dayData = new HashMap<>();
            dayData.put("date", dayStart.toLocalDate().toString());
            dayData.put("total", total);
            dayData.put("phishing", phishing);
            dayData.put("safe", total - phishing);

            trendData.add(dayData);
        }

        return trendData;
    }

    public Map<String, Long> getRiskDistribution() {
        Map<String, Long> distribution = new HashMap<>();

        Long highCount = historyRepository.countByRiskLevel(DetectionHistory.RiskLevel.HIGH);
        Long mediumCount = historyRepository.countByRiskLevel(DetectionHistory.RiskLevel.MEDIUM);
        Long lowCount = historyRepository.countByRiskLevel(DetectionHistory.RiskLevel.LOW);
        Long safeCount = historyRepository.countByRiskLevel(DetectionHistory.RiskLevel.SAFE);

        distribution.put("HIGH", highCount != null ? highCount : 0);
        distribution.put("MEDIUM", mediumCount != null ? mediumCount : 0);
        distribution.put("LOW", lowCount != null ? lowCount : 0);
        distribution.put("SAFE", safeCount != null ? safeCount : 0);

        return distribution;
    }
}
