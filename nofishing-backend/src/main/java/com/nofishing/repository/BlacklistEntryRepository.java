package com.nofishing.repository;

import com.nofishing.entity.BlacklistEntry;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface BlacklistEntryRepository extends JpaRepository<BlacklistEntry, Long> {
    List<BlacklistEntry> findByEnabledTrue();
    Optional<BlacklistEntry> findByPattern(String pattern);
    boolean existsByPattern(String pattern);
}
