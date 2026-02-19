package com.nofishing.service;

import com.nofishing.annotation.Audited;
import com.nofishing.dto.PaginatedResponse;
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
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class DetectionHistoryService {

    private final DetectionHistoryRepository repository;

    @Transactional
    public DetectionHistory save(DetectionHistory history) {
        return repository.save(history);
    }

    public PaginatedResponse<DetectionHistory> getHistory(
            String keyword,
            Boolean isPhishing,
            LocalDateTime startTime,
            LocalDateTime endTime,
            int page,
            int size) {

        Pageable pageable = PageRequest.of(page, size, Sort.by("detectedAt").descending());
        Page<DetectionHistory> historyPage;

        if (keyword != null && !keyword.isBlank()) {
            historyPage = repository.findByUrlContaining(keyword, pageable);
        } else if (isPhishing != null) {
            historyPage = repository.findByIsPhishing(isPhishing, pageable);
        } else if (startTime != null && endTime != null) {
            historyPage = repository.findByDetectedAtBetween(startTime, endTime, pageable);
        } else {
            historyPage = repository.findAll(pageable);
        }

        return PaginatedResponse.<DetectionHistory>builder()
                .content(historyPage.getContent())
                .pageNumber(historyPage.getNumber())
                .pageSize(historyPage.getSize())
                .totalElements(historyPage.getTotalElements())
                .totalPages(historyPage.getTotalPages())
                .first(historyPage.isFirst())
                .last(historyPage.isLast())
                .build();
    }

    public DetectionHistory getById(Long id) {
        return repository.findById(id)
                .orElseThrow(() -> new RuntimeException("History record not found"));
    }

    @Transactional
    @Audited(operation = "DELETE_HISTORY", module = "HISTORY", targetType = "RECORD")
    public void deleteById(Long id) {
        repository.deleteById(id);
    }

    @Transactional
    @Audited(operation = "BATCH_DELETE_HISTORY", module = "HISTORY", targetType = "BATCH")
    public void deleteOldRecords(LocalDateTime before) {
        List<DetectionHistory> oldRecords = repository.findAll().stream()
                .filter(h -> h.getDetectedAt().isBefore(before))
                .collect(Collectors.toList());
        repository.deleteAll(oldRecords);
    }

    public DetectionHistory findLatestByUrl(String url) {
        return repository.findFirstByUrlOrderByDetectedAtDesc(url).orElse(null);
    }

    /**
     * Get all history records for export (with optional filters)
     */
    public List<DetectionHistory> getAllForExport(
            String keyword,
            Boolean isPhishing,
            LocalDateTime startTime,
            LocalDateTime endTime) {

        if (keyword != null && !keyword.isBlank()) {
            return repository.findByUrlContaining(keyword);
        } else if (isPhishing != null) {
            return repository.findByIsPhishing(isPhishing);
        } else if (startTime != null && endTime != null) {
            return repository.findByDetectedAtBetween(startTime, endTime);
        } else {
            return repository.findAll();
        }
    }
}
