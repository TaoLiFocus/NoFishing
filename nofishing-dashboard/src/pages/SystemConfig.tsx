import React, { useEffect, useState } from 'react';
import {
  Card,
  Form,
  Input,
  InputNumber,
  Switch,
  Button,
  Space,
  message,
  Tabs,
  Tag,
  Descriptions,
} from 'antd';
import {
  SaveOutlined,
  ReloadOutlined,
  UndoOutlined,
  SettingOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  SecurityScanOutlined,
  ApiOutlined,
} from '@ant-design/icons';
import { systemApi, type SystemConfig } from '../api/system';
import { useRequireAuth } from '../hooks/useRequireAuth';

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
  SYSTEM: <SettingOutlined />,
  ML: <ThunderboltOutlined />,
  CACHE: <DatabaseOutlined />,
  SECURITY: <SecurityScanOutlined />,
  FEATURE: <ApiOutlined />,
};

const CATEGORY_COLORS: Record<string, string> = {
  SYSTEM: 'blue',
  ML: 'orange',
  CACHE: 'green',
  SECURITY: 'red',
  FEATURE: 'purple',
};

const SystemConfig: React.FC = () => {
  useRequireAuth(true);
  const [configs, setConfigs] = useState<Record<string, SystemConfig[]>>({});
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState<Record<string, boolean>>({});
  const [form] = Form.useForm();

  const fetchConfigs = async () => {
    setLoading(true);
    try {
      const response = await systemApi.getAllConfigs();
      const configData = response.data;

      // Auto-initialize if configs are empty
      if (Object.keys(configData).length === 0) {
        await systemApi.initializeDefaults();
        const response2 = await systemApi.getAllConfigs();
        setConfigs(response2.data);
        message.info('系统配置已初始化');
      } else {
        setConfigs(configData);
      }
    } catch (error) {
      message.error('获取系统配置失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfigs();
  }, []);

  const handleUpdate = async (configKey: string, value: any) => {
    setSaving((prev) => ({ ...prev, [configKey]: true }));
    try {
      await systemApi.updateConfig(configKey, String(value));
      message.success('配置更新成功');
      fetchConfigs();
    } catch (error) {
      message.error('配置更新失败');
    } finally {
      setSaving((prev) => ({ ...prev, [configKey]: false }));
    }
  };

  const handleReset = async () => {
    setLoading(true);
    try {
      await systemApi.resetToDefaults();
      message.success('配置已重置为默认值');
      fetchConfigs();
    } catch (error) {
      message.error('重置配置失败');
    } finally {
      setLoading(false);
    }
  };

  const renderConfigItem = (config: SystemConfig) => {
    const isNumber = ['detection.threshold', 'cache.ttl', 'cache.max-size',
                      'ml.service.timeout', 'ml.service.connect-timeout',
                      'ml.service.read-timeout'].includes(config.configKey);
    const isBoolean = ['registration.enabled', 'maintenance.mode'].includes(config.configKey);

    return (
      <div key={config.configKey} style={{ marginBottom: 16 }}>
        <Descriptions column={1} size="small">
          <Descriptions.Item label={<strong>{config.configKey}</strong>}>
            <Space>
              {isNumber ? (
                <InputNumber
                  defaultValue={parseFloat(config.configValue)}
                  min={0}
                  max={config.configKey.includes('threshold') ? 1 : 100000}
                  step={config.configKey.includes('threshold') ? 0.1 : 100}
                  style={{ width: 200 }}
                  onChange={(value) => {
                    if (value !== null) {
                      handleUpdate(config.configKey, value);
                    }
                  }}
                  loading={saving[config.configKey]}
                />
              ) : isBoolean ? (
                <Switch
                  defaultChecked={config.configValue === 'true'}
                  onChange={(checked) => handleUpdate(config.configKey, checked)}
                  loading={saving[config.configKey]}
                />
              ) : (
                <Input
                  defaultValue={config.configValue}
                  onBlur={(e) => handleUpdate(config.configKey, e.target.value)}
                  style={{ width: 300 }}
                  suffix={saving[config.configKey] && <span style={{ color: '#1890ff' }}>保存中...</span>}
                />
              )}
            </Space>
          </Descriptions.Item>
          <Descriptions.Item label="说明">{config.description}</Descriptions.Item>
        </Descriptions>
      </div>
    );
  };

  const renderCategory = (category: string, items: SystemConfig[]) => (
    <Card
      key={category}
      title={
        <Space>
          {CATEGORY_ICONS[category]}
          <span>{category}</span>
          <Tag color={CATEGORY_COLORS[category]}>{items.length} 项</Tag>
        </Space>
      }
      style={{ marginBottom: 16 }}
    >
      {items.map((item) => renderConfigItem(item))}
    </Card>
  );

  const categoryItems = Object.entries(configs).map(([category, items]) => ({
    key: category,
    label: (
      <span>
        {CATEGORY_ICONS[category]} {category}
      </span>
    ),
    children: renderCategory(category, items),
  }));

  return (
    <div>
      <Space style={{ marginBottom: 16 }}>
        <h2>系统配置管理</h2>
        <Button icon={<ReloadOutlined />} onClick={fetchConfigs} loading={loading}>
          刷新
        </Button>
        <Button icon={<UndoOutlined />} onClick={handleReset} loading={loading}>
          重置为默认值
        </Button>
      </Space>

      <Card>
        <Tabs items={categoryItems} />
      </Card>
    </div>
  );
};

export default SystemConfig;
