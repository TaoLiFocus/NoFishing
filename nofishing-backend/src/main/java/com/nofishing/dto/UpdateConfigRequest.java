package com.nofishing.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

/**
 * Request DTO for updating system configuration
 */
@Data
public class UpdateConfigRequest {

    @NotBlank(message = "Configuration value cannot be blank")
    private String value;
}
