package com.nofishing.repository;

import com.nofishing.entity.WhitelistEntry;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface WhitelistEntryRepository extends JpaRepository<WhitelistEntry, Long> {
    List<WhitelistEntry> findByEnabledTrue();
    Optional<WhitelistEntry> findByPattern(String pattern);
    boolean existsByPattern(String pattern);
}
