package com.nofishing.service;

import com.nofishing.annotation.Audited;
import com.nofishing.entity.BlacklistEntry;
import com.nofishing.repository.BlacklistEntryRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Lazy;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

@Slf4j
@Service
@RequiredArgsConstructor
public class BlacklistService {

    private final BlacklistEntryRepository repository;
    @Lazy
    private final WhitelistService whitelistService;

    @Transactional
    @Audited(operation = "ADD_BLACKLIST", module = "BLACKLIST", targetType = "DOMAIN")
    public BlacklistEntry create(BlacklistEntry entry) {
        // Check if pattern already exists in blacklist
        if (repository.existsByPattern(entry.getPattern())) {
            throw new RuntimeException("Pattern already exists in blacklist");
        }

        // Check if pattern exists in whitelist - if so, remove it for mutual exclusion
        if (whitelistService.existsByPattern(entry.getPattern())) {
            log.info("Pattern {} exists in whitelist, removing for mutual exclusion", entry.getPattern());
            whitelistService.deleteByPattern(entry.getPattern());
        }

        return repository.save(entry);
    }

    @Transactional
    @Audited(operation = "UPDATE_BLACKLIST", module = "BLACKLIST", targetType = "DOMAIN")
    public BlacklistEntry update(Long id, BlacklistEntry entry) {
        BlacklistEntry existing = repository.findById(id)
                .orElseThrow(() -> new RuntimeException("Entry not found"));

        if (!existing.getPattern().equals(entry.getPattern())
                && repository.existsByPattern(entry.getPattern())) {
            throw new RuntimeException("Pattern already exists");
        }

        existing.setPattern(entry.getPattern());
        existing.setType(entry.getType());
        existing.setEnabled(entry.getEnabled());
        existing.setComment(entry.getComment());
        existing.setThreatType(entry.getThreatType());
        existing.setExpiresAt(entry.getExpiresAt());

        return repository.save(existing);
    }

    @Transactional
    @Audited(operation = "DELETE_BLACKLIST", module = "BLACKLIST", targetType = "DOMAIN")
    public void deleteById(Long id) {
        repository.deleteById(id);
    }

    public BlacklistEntry getById(Long id) {
        return repository.findById(id)
                .orElseThrow(() -> new RuntimeException("Entry not found"));
    }

    public Page<BlacklistEntry> findAll(int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
        return repository.findAll(pageable);
    }

    public List<BlacklistEntry> findAllEnabled() {
        return repository.findByEnabledTrue();
    }

    public boolean isBlacklisted(String url) {
        List<BlacklistEntry> entries = findAllEnabled();
        for (BlacklistEntry entry : entries) {
            if (matchesPattern(url, entry.getPattern())) {
                return true;
            }
        }
        return false;
    }

    private boolean matchesPattern(String url, String pattern) {
        try {
            // Handle wildcard patterns
            // *://*google.com* should match https://www.google.com, http://google.com, etc.
            // *google.com* should match any URL containing google.com

            // Escape special regex characters except * and ?
            String regex = pattern
                    .replaceAll("([.+^${}()|\\[\\]\\\\])", "\\\\$1")
                    .replace("*", ".*")
                    .replace("?", ".");

            // If pattern doesn't start with *, match from beginning
            if (!pattern.startsWith("*")) {
                regex = "^" + regex;
            }
            // If pattern doesn't end with *, match till end
            if (!pattern.endsWith("*")) {
                regex = regex + "$";
            }

            return Pattern.matches(regex, url);
        } catch (PatternSyntaxException e) {
            log.error("Invalid pattern: {}", pattern, e);
            return false;
        }
    }

    public boolean existsByPattern(String pattern) {
        return repository.existsByPattern(pattern);
    }

    /**
     * Delete blacklist entry by pattern (for mutual exclusion with whitelist)
     */
    @Transactional
    @Audited(operation = "DELETE_BLACKLIST", module = "BLACKLIST", targetType = "DOMAIN")
    public void deleteByPattern(String pattern) {
        repository.deleteByPattern(pattern);
        log.info("Deleted blacklist entry with pattern: {}", pattern);
    }

    /**
     * Batch delete blacklist entries
     */
    @Transactional
    @Audited(operation = "BATCH_DELETE", module = "BLACKLIST", targetType = "BATCH")
    public void batchDelete(List<Long> ids) {
        log.info("Batch deleting {} blacklist entries", ids.size());
        repository.deleteAllById(ids);
    }

    /**
     * Batch import blacklist entries from list of patterns
     */
    @Transactional
    @Audited(operation = "BATCH_IMPORT", module = "BLACKLIST", targetType = "BATCH")
    public List<String> batchImport(List<String> patterns, String addedBy, String comment) {
        List<String> results = new ArrayList<>();
        int success = 0;
        int skipped = 0;

        for (String pattern : patterns) {
            pattern = pattern.trim();
            if (pattern.isEmpty()) {
                continue;
            }

            if (repository.existsByPattern(pattern)) {
                skipped++;
                results.add("SKIPPED: " + pattern + " (already exists)");
            } else {
                BlacklistEntry entry = new BlacklistEntry();
                entry.setPattern(pattern);
                entry.setEnabled(true);
                entry.setAddedBy(addedBy);
                entry.setComment(comment);
                repository.save(entry);
                success++;
                results.add("ADDED: " + pattern);
            }
        }

        log.info("Batch import completed: {} added, {} skipped", success, skipped);
        return results;
    }
}
