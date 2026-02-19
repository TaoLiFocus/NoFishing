package com.nofishing.annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Annotation for marking methods that should be audited
 * Used by AOP aspect to automatically log operations
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface Audited {

    /**
     * Operation type (e.g., CREATE_USER, DELETE_USER)
     */
    String operation();

    /**
     * Module name (e.g., USER, WHITELIST, BLACKLIST)
     */
    String module();

    /**
     * Target type (e.g., USER, DOMAIN, RECORD)
     */
    String targetType() default "UNKNOWN";

    /**
     * Whether to log request parameters
     */
    boolean logParams() default true;

    /**
     * Whether to log return value
     */
    boolean logResult() default false;

    /**
     * Description of the operation
     */
    String description() default "";
}
