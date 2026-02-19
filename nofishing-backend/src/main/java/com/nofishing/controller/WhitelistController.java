package com.nofishing.controller;

import com.nofishing.entity.WhitelistEntry;
import com.nofishing.service.WhitelistService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/whitelist")
@RequiredArgsConstructor
public class WhitelistController {

    private final WhitelistService service;

    @GetMapping
    public ResponseEntity<Page<WhitelistEntry>> findAll(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        Page<WhitelistEntry> entries = service.findAll(page, size);
        return ResponseEntity.ok(entries);
    }

    @GetMapping("/{id}")
    public ResponseEntity<WhitelistEntry> getById(@PathVariable Long id) {
        WhitelistEntry entry = service.getById(id);
        return ResponseEntity.ok(entry);
    }

    @PostMapping
    public ResponseEntity<WhitelistEntry> create(@Valid @RequestBody WhitelistEntry entry) {
        WhitelistEntry created = service.create(entry);
        return ResponseEntity.ok(created);
    }

    @PutMapping("/{id}")
    public ResponseEntity<WhitelistEntry> update(@PathVariable Long id, @Valid @RequestBody WhitelistEntry entry) {
        WhitelistEntry updated = service.update(id, entry);
        return ResponseEntity.ok(updated);
    }

    @DeleteMapping("/{id}")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<Void> deleteById(@PathVariable Long id) {
        service.deleteById(id);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/check")
    public ResponseEntity<Map<String, Boolean>> checkUrl(@RequestParam String url) {
        boolean isWhitelisted = service.isWhitelisted(url);
        return ResponseEntity.ok(Map.of("whitelisted", isWhitelisted));
    }

    @GetMapping("/exists")
    public ResponseEntity<Map<String, Boolean>> existsByPattern(@RequestParam String pattern) {
        boolean exists = service.existsByPattern(pattern);
        return ResponseEntity.ok(Map.of("exists", exists));
    }

    /**
     * Batch delete whitelist entries (Admin only)
     */
    @DeleteMapping("/batch")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<Map<String, Object>> batchDelete(@RequestBody List<Long> ids) {
        service.batchDelete(ids);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "deleted", ids.size()
        ));
    }

    /**
     * Batch import whitelist entries (Admin only)
     */
    @PostMapping("/batch-import")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<Map<String, Object>> batchImport(
            @RequestBody Map<String, Object> request,
            org.springframework.security.core.Authentication authentication) {
        @SuppressWarnings("unchecked")
        List<String> patterns = (List<String>) request.get("patterns");
        String comment = (String) request.getOrDefault("comment", "Batch imported");
        String addedBy = authentication.getName();

        List<String> results = service.batchImport(patterns, addedBy, comment);

        return ResponseEntity.ok(Map.of(
                "success", true,
                "results", results,
                "total", results.size()
        ));
    }
}
