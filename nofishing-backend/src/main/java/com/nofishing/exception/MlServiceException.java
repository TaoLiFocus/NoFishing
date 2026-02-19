package com.nofishing.exception;

/**
 * Exception thrown when ML service communication fails
 *
 * @author NoFishing Team
 */
public class MlServiceException extends RuntimeException {

    public MlServiceException(String message) {
        super(message);
    }

    public MlServiceException(String message, Throwable cause) {
        super(message, cause);
    }
}
