package com.nofishing.repository;

import com.nofishing.entity.ApiKey;
import com.nofishing.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface ApiKeyRepository extends JpaRepository<ApiKey, Long> {

    Optional<ApiKey> findByKeyValue(String keyValue);

    List<ApiKey> findByUserId(Long userId);

    List<ApiKey> findByUserIdAndIsEnabledTrue(Long userId);

    List<ApiKey> findByIsEnabledTrueAndExpiresAtAfter(LocalDateTime now);

    List<ApiKey> findByExpiresAtBefore(LocalDateTime date);

    boolean existsByKeyValue(String keyValue);

    void deleteByUserId(Long userId);

    /**
     * Find all API keys with user relationship eagerly loaded
     */
    @Query("SELECT ak FROM ApiKey ak LEFT JOIN FETCH ak.user")
    List<ApiKey> findAllWithUser();

    /**
     * Find API keys by user ID with user relationship eagerly loaded
     */
    @Query("SELECT ak FROM ApiKey ak LEFT JOIN FETCH ak.user WHERE ak.user.id = :userId")
    List<ApiKey> findByUserIdWithUser(@org.springframework.data.repository.query.Param("userId") Long userId);
}
