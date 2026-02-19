package com.nofishing.util;

import com.nofishing.entity.DetectionHistory;
import lombok.extern.slf4j.Slf4j;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.springframework.stereotype.Component;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Date;
import java.util.List;

/**
 * Utility class for generating Excel files
 */
@Slf4j
@Component
public class ExcelGenerator {

    /**
     * Generate Excel file for detection history
     */
    public byte[] generateDetectionHistory(List<DetectionHistory> data) throws IOException {
        try (Workbook workbook = new XSSFWorkbook()) {
            Sheet sheet = workbook.createSheet("Detection History");

            // Create header style
            CellStyle headerStyle = createHeaderStyle(workbook);

            // Create data style
            CellStyle dataStyle = createDataStyle(workbook);

            // Create header row
            Row headerRow = sheet.createRow(0);
            String[] headers = {
                    "ID", "URL", "Is Phishing", "Confidence", "Risk Level",
                    "Source", "Detected At", "Processing Time (ms)", "IP Address"
            };

            for (int i = 0; i < headers.length; i++) {
                Cell cell = headerRow.createCell(i);
                cell.setCellValue(headers[i]);
                cell.setCellStyle(headerStyle);
            }

            // Auto-size columns
            for (int i = 0; i < headers.length; i++) {
                sheet.autoSizeColumn(i);
            }

            // Create data rows
            int rowNum = 1;
            for (DetectionHistory history : data) {
                Row row = sheet.createRow(rowNum++);

                createCell(row, 0, history.getId().toString(), dataStyle);
                createCell(row, 1, history.getUrl(), dataStyle);
                createCell(row, 2, history.getIsPhishing() != null ? history.getIsPhishing().toString() : "N/A", dataStyle);
                createCell(row, 3, history.getConfidence() != null ? String.format("%.4f", history.getConfidence()) : "N/A", dataStyle);
                createCell(row, 4, history.getRiskLevel() != null ? history.getRiskLevel().name() : "N/A", dataStyle);
                createCell(row, 5, history.getSource() != null ? history.getSource() : "N/A", dataStyle);
                createCell(row, 6, history.getDetectedAt() != null ? history.getDetectedAt().toString() : "N/A", dataStyle);
                createCell(row, 7, history.getProcessingTimeMs() != null ? history.getProcessingTimeMs().toString() : "N/A", dataStyle);
                createCell(row, 8, history.getIpAddress() != null ? history.getIpAddress() : "N/A", dataStyle);
            }

            // Write to byte array
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            workbook.write(outputStream);

            log.info("Generated Excel file with {} rows", data.size());
            return outputStream.toByteArray();
        }
    }

    /**
     * Create header style
     */
    private CellStyle createHeaderStyle(Workbook workbook) {
        CellStyle style = workbook.createCellStyle();

        // Font
        Font font = workbook.createFont();
        font.setBold(true);
        font.setFontHeightInPoints((short) 11);
        style.setFont(font);

        // Background color
        style.setFillForegroundColor(IndexedColors.GREY_25_PERCENT.getIndex());
        style.setFillPattern(FillPatternType.SOLID_FOREGROUND);

        // Border
        style.setBorderBottom(BorderStyle.THIN);
        style.setBorderTop(BorderStyle.THIN);
        style.setBorderLeft(BorderStyle.THIN);
        style.setBorderRight(BorderStyle.THIN);

        // Alignment
        style.setAlignment(HorizontalAlignment.CENTER);
        style.setVerticalAlignment(VerticalAlignment.CENTER);

        return style;
    }

    /**
     * Create data style
     */
    private CellStyle createDataStyle(Workbook workbook) {
        CellStyle style = workbook.createCellStyle();

        // Border
        style.setBorderBottom(BorderStyle.THIN);
        style.setBorderTop(BorderStyle.THIN);
        style.setBorderLeft(BorderStyle.THIN);
        style.setBorderRight(BorderStyle.THIN);

        // Alignment
        style.setVerticalAlignment(VerticalAlignment.CENTER);

        return style;
    }

    /**
     * Create cell with style
     */
    private void createCell(Row row, int column, String value, CellStyle style) {
        Cell cell = row.createCell(column);
        cell.setCellValue(value);
        cell.setCellStyle(style);
    }

    /**
     * Generate Excel file for generic list data
     */
    public byte[] generateGenericList(String sheetName, List<String> headers, List<List<String>> data) throws IOException {
        try (Workbook workbook = new XSSFWorkbook()) {
            Sheet sheet = workbook.createSheet(sheetName);

            // Create header style
            CellStyle headerStyle = createHeaderStyle(workbook);
            CellStyle dataStyle = createDataStyle(workbook);

            // Create header row
            Row headerRow = sheet.createRow(0);
            for (int i = 0; i < headers.size(); i++) {
                Cell cell = headerRow.createCell(i);
                cell.setCellValue(headers.get(i));
                cell.setCellStyle(headerStyle);
            }

            // Auto-size columns
            for (int i = 0; i < headers.size(); i++) {
                sheet.autoSizeColumn(i);
            }

            // Create data rows
            int rowNum = 1;
            for (List<String> rowData : data) {
                Row row = sheet.createRow(rowNum++);
                for (int i = 0; i < rowData.size(); i++) {
                    createCell(row, i, rowData.get(i), dataStyle);
                }
            }

            // Write to byte array
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            workbook.write(outputStream);

            return outputStream.toByteArray();
        }
    }
}
