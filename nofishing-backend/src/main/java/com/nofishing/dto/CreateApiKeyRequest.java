package com.nofishing.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Request DTO for creating an API key
 */
@Data
public class CreateApiKeyRequest {

    @NotBlank(message = "Name is required")
    @Size(min = 1, max = 100, message = "Name must be between 1 and 100 characters")
    private String name;

    @NotNull(message = "User ID is required")
    private Long userId;

    @NotNull(message = "Permissions are required")
    private List<String> permissions;

    private LocalDateTime expiresAt;
}
