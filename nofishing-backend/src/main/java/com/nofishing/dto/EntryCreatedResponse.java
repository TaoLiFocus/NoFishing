package com.nofishing.dto;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import com.nofishing.entity.WhitelistEntry;
import com.nofishing.entity.BlacklistEntry;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Response DTO for whitelist/blacklist entry creation
 * Includes the created entry and optional message (e.g., about mutual exclusion)
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(PropertyNamingStrategies.LowerCamelCaseStrategy.class)
public class EntryCreatedResponse {

    private Long id;
    private String pattern;
    private String type;  // "whitelist" or "blacklist"
    private String message;  // Optional message about the operation

    public static EntryCreatedResponse fromWhitelistEntry(WhitelistEntry entry, String message) {
        return EntryCreatedResponse.builder()
                .id(entry.getId())
                .pattern(entry.getPattern())
                .type("whitelist")
                .message(message)
                .build();
    }

    public static EntryCreatedResponse fromBlacklistEntry(BlacklistEntry entry, String message) {
        return EntryCreatedResponse.builder()
                .id(entry.getId())
                .pattern(entry.getPattern())
                .type("blacklist")
                .message(message)
                .build();
    }
}
