import React, { useState } from 'react';
import { Card, Form, Input, Button, Alert, Descriptions, Tag, Space, Divider } from 'antd';
import { SafetyOutlined, ClearOutlined } from '@ant-design/icons';
import { detectionApi } from '../api/detection';
import type { DetectionResponse } from '../types';
import { useRequireAuth } from '../hooks/useRequireAuth';

const Detection: React.FC = () => {
  useRequireAuth();
  const [form] = Form.useForm();
  const [detecting, setDetecting] = useState(false);
  const [result, setResult] = useState<DetectionResponse | null>(null);

  const handleDetect = async (values: { url: string }) => {
    setDetecting(true);
    setResult(null);
    try {
      const response = await detectionApi.detect({ url: values.url });
      setResult(response.data);
    } catch (error) {
      console.error('Detection failed:', error);
    } finally {
      setDetecting(false);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'HIGH': return 'red';
      case 'MEDIUM': return 'orange';
      case 'LOW': return 'gold';
      case 'SAFE': return 'green';
      default: return 'default';
    }
  };

  const getRiskTag = (level: string) => {
    switch (level) {
      case 'HIGH': return '钓鱼网站 - 高风险';
      case 'MEDIUM': return '可疑';
      case 'LOW': return '低风险';
      case 'SAFE': return '安全';
      default: return level;
    }
  };

  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ marginBottom: 32, fontSize: 28, fontWeight: 600 }}>URL检测</h1>

      <Card>
        <Form
          form={form}
          layout="vertical"
          onFinish={handleDetect}
        >
          <Form.Item
            name="url"
            label="要检测的URL"
            rules={[
              { required: true, message: '请输入URL' },
              { type: 'url', message: '请输入有效的URL' },
            ]}
          >
            <Input
              placeholder="https://example.com"
              prefix={<SafetyOutlined />}
              size="large"
              allowClear
            />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                loading={detecting}
                size="large"
                icon={<SafetyOutlined />}
              >
                检测
              </Button>
              <Button
                onClick={() => {
                  form.resetFields();
                  setResult(null);
                }}
                size="large"
                icon={<ClearOutlined />}
              >
                清空
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      {result && (
        <Card style={{ marginTop: 32 }} title="检测结果">
          <Alert
            message={result.isPhishing ? '警告：检测到钓鱼网站！' : '安全：未检测到钓鱼网站'}
            description={
              result.isPhishing
                ? `该URL显示钓鱼网站特征，置信度为 ${Math.round((result.confidence || 0) * 100)}%。`
                : '该URL看起来是安全的。'
            }
            type={result.isPhishing ? 'error' : 'success'}
            showIcon
            style={{ marginBottom: 16 }}
          />

          <Descriptions bordered column={2} size="small">
            <Descriptions.Item label="URL">{result.url}</Descriptions.Item>
            <Descriptions.Item label="风险等级">
              <Tag color={getRiskColor(result.riskLevel)}>{getRiskTag(result.riskLevel)}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="置信度">
              {Math.round((result.confidence || 0) * 100)}%
            </Descriptions.Item>
            <Descriptions.Item label="检测时间">
              {new Date(result.timestamp).toLocaleString()}
            </Descriptions.Item>
          </Descriptions>

          {result.features && Object.keys(result.features).length > 0 && (
            <>
              <Divider>特征</Divider>
              <div style={{ maxHeight: 200, overflowY: 'auto' }}>
                {Object.entries(result.features).map(([key, value]) => (
                  <div key={key} style={{ marginBottom: 8 }}>
                    <strong>{key}:</strong> {typeof value === 'number' ? value.toFixed(4) : String(value)}
                  </div>
                ))}
              </div>
            </>
          )}
        </Card>
      )}

      <Card style={{ marginTop: 32 }} title="示例">
        <p style={{ color: '#888', marginBottom: 16, fontSize: 14, lineHeight: 1.6 }}>
          试试这些示例URL：
        </p>
        <Space direction="vertical" style={{ width: '100%' }}>
          <a onClick={() => form.setFieldValue('url', 'http://example.com')}>http://example.com</a>
          <a onClick={() => form.setFieldValue('url', 'http://paypal-security-center.com')}>http://paypal-security-center.com</a>
          <a onClick={() => form.setFieldValue('url', 'http://apple-support-id.net')}>http://apple-support-id.net</a>
        </Space>
      </Card>
    </div>
  );
};

export default Detection;
