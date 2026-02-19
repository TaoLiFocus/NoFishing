import React, { useEffect, useState } from 'react';
import {
  Table,
  Card,
  Form,
  Select,
  DatePicker,
  Button,
  Space,
  Tag,
  Input,
} from 'antd';
import {
  SearchOutlined,
  ReloadOutlined,
  FileTextOutlined,
  ClockCircleOutlined,
  UserOutlined,
  AppstoreOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';
import { auditApi, type AuditLog } from '../api/audit';
import { useRequireAuth } from '../hooks/useRequireAuth';

const { RangePicker } = DatePicker;

// Chinese translations for operations
const OPERATION_LABELS: Record<string, string> = {
  LOGIN: '用户登录',
  LOGOUT: '用户登出',
  CREATE_USER: '创建用户',
  UPDATE_USER: '更新用户',
  DELETE_USER: '删除用户',
  RESET_PASSWORD: '重置密码',
  CHANGE_PASSWORD: '修改密码',
  ADD_WHITELIST: '添加白名单',
  DELETE_WHITELIST: '删除白名单',
  UPDATE_WHITELIST: '更新白名单',
  ADD_BLACKLIST: '添加黑名单',
  DELETE_BLACKLIST: '删除黑名单',
  UPDATE_BLACKLIST: '更新黑名单',
  DELETE_HISTORY: '删除历史记录',
  BATCH_DELETE_HISTORY: '批量删除历史',
  UPDATE_CONFIG: '更新系统配置',
  RESET_CONFIG: '重置系统配置',
  CREATE_API_KEY: '创建API密钥',
  REVOKE_API_KEY: '撤销API密钥',
  UPDATE_API_KEY: '更新API密钥',
  BATCH_IMPORT: '批量导入',
  BATCH_DELETE: '批量删除',
};

// Chinese translations for modules
const MODULE_LABELS: Record<string, string> = {
  AUTH: '认证',
  USER: '用户',
  WHITELIST: '白名单',
  BLACKLIST: '黑名单',
  DETECTION: '检测',
  HISTORY: '历史记录',
  SYSTEM: '系统',
  API_KEY: 'API密钥',
};

const OPERATION_COLORS: Record<string, string> = {
  LOGIN: 'green',
  LOGOUT: 'default',
  CREATE_USER: 'blue',
  UPDATE_USER: 'cyan',
  DELETE_USER: 'red',
  RESET_PASSWORD: 'orange',
  ADD_WHITELIST: 'lime',
  DELETE_WHITELIST: 'magenta',
  ADD_BLACKLIST: 'volcano',
  DELETE_BLACKLIST: 'red',
  DELETE_HISTORY: 'purple',
  BATCH_DELETE_HISTORY: 'purple',
  UPDATE_CONFIG: 'gold',
  CREATE_API_KEY: 'blue',
  REVOKE_API_KEY: 'red',
  UPDATE_API_KEY: 'cyan',
  BATCH_IMPORT: 'green',
  BATCH_DELETE: 'red',
};

const MODULE_COLORS: Record<string, string> = {
  AUTH: 'blue',
  USER: 'green',
  WHITELIST: 'lime',
  BLACKLIST: 'red',
  DETECTION: 'orange',
  HISTORY: 'purple',
  SYSTEM: 'gold',
  API_KEY: 'cyan',
};

const AuditLogPage: React.FC = () => {
  useRequireAuth(true);
  const [data, setData] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(false);
  const [pagination, setPagination] = useState({ current: 1, pageSize: 20, total: 0 });
  const [form] = Form.useForm();

  const fetchLogs = async (page = 1, size = 20, filters?: any) => {
    setLoading(true);
    try {
      const params: any = { page: page - 1, size };

      if (filters?.operation) params.operation = filters.operation;
      if (filters?.module) params.module = filters.module;
      if (filters?.operatedBy) params.operatedBy = filters.operatedBy;
      if (filters?.dateRange) {
        params.startDate = filters.dateRange[0].format('YYYY-MM-DDTHH:mm:ss');
        params.endDate = filters.dateRange[1].format('YYYY-MM-DDTHH:mm:ss');
      }

      const response = await auditApi.searchLogs(params);
      setData(response.data.content);
      setPagination({
        current: page,
        pageSize: size,
        total: response.data.totalElements,
      });
    } catch (error) {
      console.error('获取审计日志失败', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  const handleSearch = () => {
    const values = form.getFieldsValue();
    fetchLogs(1, pagination.pageSize, values);
  };

  const handleReset = () => {
    form.resetFields();
    fetchLogs();
  };

  const getOperationLabel = (operation: string) => {
    return OPERATION_LABELS[operation] || operation;
  };

  const getModuleLabel = (module: string) => {
    return MODULE_LABELS[module] || module;
  };

  const columns: ColumnsType<AuditLog> = [
    {
      title: '时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      width: 180,
      render: (date: string) => (
        <Space>
          <ClockCircleOutlined />
          {dayjs(date).format('YYYY-MM-DD HH:mm:ss')}
        </Space>
      ),
      sorter: true,
    },
    {
      title: '操作',
      dataIndex: 'operation',
      key: 'operation',
      width: 150,
      render: (operation: string) => (
        <Tag color={OPERATION_COLORS[operation] || 'default'}>{getOperationLabel(operation)}</Tag>
      ),
    },
    {
      title: '模块',
      dataIndex: 'module',
      key: 'module',
      width: 120,
      render: (module: string) => (
        <Tag icon={<AppstoreOutlined />} color={MODULE_COLORS[module] || 'default'}>{getModuleLabel(module)}</Tag>
      ),
    },
    {
      title: '操作者',
      dataIndex: 'operatedBy',
      key: 'operatedBy',
      width: 120,
      render: (user: string) => (
        <Space>
          <UserOutlined />
          {user}
        </Space>
      ),
    },
    {
      title: '目标类型',
      dataIndex: 'targetType',
      key: 'targetType',
      width: 100,
      render: (type: string) => type && <Tag>{type}</Tag>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => {
        const color = status === 'SUCCESS' ? 'green' : status === 'FAILURE' ? 'red' : 'orange';
        const label = status === 'SUCCESS' ? '成功' : status === 'FAILURE' ? '失败' : status;
        return <Tag color={color}>{label}</Tag>;
      },
    },
    {
      title: '详情',
      dataIndex: 'targetValue',
      key: 'targetValue',
      ellipsis: true,
      render: (value: string) => value && <span title={value}>{value.substring(0, 50)}...</span>,
    },
  ];

  return (
    <div>
      <Space style={{ marginBottom: 16 }}>
        <FileTextOutlined />
        <h2>审计日志</h2>
      </Space>

      <Card style={{ marginBottom: 16 }}>
        <Form form={form} layout="inline">
          <Form.Item name="operation" label="操作类型">
            <Select style={{ width: 150 }} placeholder="选择操作" allowClear>
              {Object.keys(OPERATION_LABELS).map((op) => (
                <Select.Option key={op} value={op}>{OPERATION_LABELS[op]}</Select.Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item name="module" label="模块">
            <Select style={{ width: 120 }} placeholder="选择模块" allowClear>
              {Object.keys(MODULE_LABELS).map((mod) => (
                <Select.Option key={mod} value={mod}>{MODULE_LABELS[mod]}</Select.Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item name="operatedBy" label="操作者">
            <Input placeholder="输入用户名" style={{ width: 120 }} />
          </Form.Item>

          <Form.Item name="dateRange" label="时间范围">
            <RangePicker showTime style={{ width: 350 }} />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" icon={<SearchOutlined />} onClick={handleSearch}>
                搜索
              </Button>
              <Button onClick={handleReset}>重置</Button>
              <Button icon={<ReloadOutlined />} onClick={() => fetchLogs()}>
                刷新
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Card>
        <Table
          columns={columns}
          dataSource={data}
          loading={loading}
          rowKey="id"
          scroll={{ x: 1200 }}
          pagination={{
            ...pagination,
            showSizeChanger: true,
            showTotal: (total) => `共 ${total} 条记录`,
            onChange: (page, pageSize) => fetchLogs(page, pageSize, form.getFieldsValue()),
          }}
        />
      </Card>
    </div>
  );
};

export default AuditLogPage;
