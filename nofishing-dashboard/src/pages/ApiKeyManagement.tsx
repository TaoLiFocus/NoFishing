import React, { useEffect, useState } from 'react';
import {
  Table,
  Card,
  Button,
  Space,
  Modal,
  Form,
  Input,
  Select,
  Tag,
  message,
  Popconfirm,
  Tooltip,
  Switch,
  DatePicker,
} from 'antd';
import {
  PlusOutlined,
  CopyOutlined,
  DeleteOutlined,
  StopOutlined,
  ReloadOutlined,
  KeyOutlined,
  ClockCircleOutlined,
  UserOutlined,
  ApiOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';
import { apiKeyApi, type ApiKey } from '../api/apiKey';
import { userApi } from '../api/auth';
import { useRequireAuth } from '../hooks/useRequireAuth';

const PERMISSION_OPTIONS = [
  { label: 'URL检测', value: 'DETECT' },
  { label: '查看历史', value: 'HISTORY_READ' },
  { label: '删除历史', value: 'HISTORY_DELETE' },
  { label: '查看白名单', value: 'WHITELIST_READ' },
  { label: '管理白名单', value: 'WHITELIST_WRITE' },
  { label: '查看黑名单', value: 'BLACKLIST_READ' },
  { label: '管理黑名单', value: 'BLACKLIST_WRITE' },
  { label: '查看统计', value: 'STATS_READ' },
];

const ApiKeyManagement: React.FC = () => {
  useRequireAuth(true);
  const [data, setData] = useState<ApiKey[]>([]);
  const [users, setUsers] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [form] = Form.useForm();

  const fetchKeys = async () => {
    setLoading(true);
    try {
      const response = await apiKeyApi.getAllKeys();
      setData(response.data);
    } catch (error) {
      message.error('获取API密钥失败');
    } finally {
      setLoading(false);
    }
  };

  const fetchUsers = async () => {
    try {
      const response = await userApi.getUsers({ page: 0, size: 1000 });
      setUsers(response.data.content);
    } catch (error) {
      console.error('获取用户列表失败', error);
    }
  };

  useEffect(() => {
    fetchKeys();
    fetchUsers();
  }, []);

  const handleCreate = () => {
    form.resetFields();
    setCreatedKey(null);
    setModalVisible(true);
  };

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      // Convert dayjs object to ISO string for expiresAt
      const requestData = {
        ...values,
        expiresAt: values.expiresAt ? values.expiresAt.toISOString() : undefined,
      };
      const response = await apiKeyApi.createKey(requestData);
      setCreatedKey(response.data.keyValue);
      message.success('API密钥创建成功');
      fetchKeys();
    } catch (error) {
      console.error('创建API密钥失败:', error);
      message.error('创建API密钥失败');
    }
  };

  const handleCopy = (keyValue: string) => {
    navigator.clipboard.writeText(keyValue);
    message.success('API密钥已复制到剪贴板');
  };

  const handleDisable = async (id: number) => {
    try {
      await apiKeyApi.disableKey(id);
      message.success('API密钥已禁用');
      fetchKeys();
    } catch (error) {
      message.error('禁用API密钥失败');
    }
  };

  const handleRevoke = async (id: number) => {
    try {
      await apiKeyApi.revokeKey(id);
      message.success('API密钥已删除');
      fetchKeys();
    } catch (error) {
      message.error('删除API密钥失败');
    }
  };

  const columns: ColumnsType<ApiKey> = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record) => (
        <Space>
          <KeyOutlined />
          <strong>{name}</strong>
        </Space>
      ),
    },
    {
      title: 'API密钥',
      dataIndex: 'keyValue',
      key: 'keyValue',
      width: 200,
      render: (keyValue: string) => (
        <Space>
          <Tooltip title="点击复制">
            <Tag
              icon={<CopyOutlined />}
              style={{ cursor: 'pointer' }}
              onClick={() => handleCopy(keyValue)}
            >
              {keyValue.substring(0, 12)}...
            </Tag>
          </Tooltip>
        </Space>
      ),
    },
    {
      title: '用户',
      dataIndex: 'username',
      key: 'username',
      width: 120,
      render: (username: string) => (
        <Space>
          <UserOutlined />
          {username}
        </Space>
      ),
    },
    {
      title: '权限',
      dataIndex: 'permissions',
      key: 'permissions',
      render: (permissions: string[]) => (
        <Space size={4} wrap>
          {permissions?.map((perm) => (
            <Tag key={perm} color="blue">{perm}</Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '过期时间',
      dataIndex: 'expiresAt',
      key: 'expiresAt',
      width: 180,
      render: (date: string) => (
        <Space>
          <ClockCircleOutlined />
          {date ? dayjs(date).format('YYYY-MM-DD HH:mm') : <Tag color="green">永久</Tag>}
        </Space>
      ),
    },
    {
      title: '最后使用',
      dataIndex: 'lastUsedAt',
      key: 'lastUsedAt',
      width: 180,
      render: (date: string) => (
        <Space>
          <ClockCircleOutlined />
          {date ? dayjs(date).format('YYYY-MM-DD HH:mm') : <Tag color="default">从未使用</Tag>}
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'isEnabled',
      key: 'isEnabled',
      width: 100,
      render: (enabled: boolean) => (
        <Tag color={enabled ? 'green' : 'red'} icon={enabled ? <ApiOutlined /> : <StopOutlined />}>
          {enabled ? '启用' : '禁用'}
        </Tag>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 150,
      fixed: 'right',
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="复制密钥">
            <Button
              type="link"
              size="small"
              icon={<CopyOutlined />}
              onClick={() => handleCopy(record.keyValue)}
            />
          </Tooltip>
          <Popconfirm
            title="确定要禁用此密钥吗？"
            onConfirm={() => handleDisable(record.id)}
            disabled={!record.isEnabled}
          >
            <Button
              type="link"
              size="small"
              icon={<StopOutlined />}
              disabled={!record.isEnabled}
            >
              禁用
            </Button>
          </Popconfirm>
          <Popconfirm
            title="确定要删除此密钥吗？"
            description="删除后无法恢复"
            onConfirm={() => handleRevoke(record.id)}
          >
            <Button
              type="link"
              size="small"
              danger
              icon={<DeleteOutlined />}
            >
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Space style={{ marginBottom: 16 }}>
        <KeyOutlined />
        <h2>API密钥管理</h2>
      </Space>

      <Card>
        <Space style={{ marginBottom: 16 }}>
          <Button type="primary" icon={<PlusOutlined />} onClick={handleCreate}>
            创建API密钥
          </Button>
          <Button icon={<ReloadOutlined />} onClick={fetchKeys} loading={loading}>
            刷新
          </Button>
        </Space>

        <Table
          columns={columns}
          dataSource={data}
          loading={loading}
          rowKey="id"
          scroll={{ x: 1400 }}
          pagination={{
            showSizeChanger: true,
            showTotal: (total) => `共 ${total} 个密钥`,
          }}
        />
      </Card>

      <Modal
        title="创建API密钥"
        open={modalVisible}
        onOk={handleSubmit}
        onCancel={() => setModalVisible(false)}
        width={600}
        okText="创建"
        cancelText="取消"
      >
        {createdKey ? (
          <div>
            <p>API密钥已创建成功，请立即复制保存，关闭窗口后将无法再次查看完整密钥：</p>
            <Input.TextArea
              value={createdKey}
              readOnly
              rows={2}
              style={{ marginBottom: 16, fontFamily: 'monospace' }}
            />
            <Button
              type="primary"
              icon={<CopyOutlined />}
              onClick={() => handleCopy(createdKey)}
              block
            >
              复制API密钥
            </Button>
          </div>
        ) : (
          <Form form={form} layout="vertical">
            <Form.Item
              name="name"
              label="密钥名称"
              rules={[{ required: true, message: '请输入密钥名称' }]}
            >
              <Input placeholder="例如：生产环境API密钥" />
            </Form.Item>

            <Form.Item
              name="userId"
              label="关联用户"
              rules={[{ required: true, message: '请选择用户' }]}
            >
              <Select
                placeholder="选择用户"
                showSearch
                optionFilterProp="children"
              >
                {users.map((user) => (
                  <Select.Option key={user.id} value={user.id}>
                    {user.username} ({user.email || '无邮箱'})
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item
              name="permissions"
              label="权限"
              rules={[{ required: true, message: '请选择权限' }]}
            >
              <Select
                mode="multiple"
                placeholder="选择权限"
                options={PERMISSION_OPTIONS}
              />
            </Form.Item>

            <Form.Item
              name="expiresAt"
              label="过期时间"
            >
              <DatePicker
                showTime
                style={{ width: '100%' }}
                placeholder="选择过期时间（默认1年）"
                disabledDate={(current) => current && current.isBefore(dayjs(), 'day')}
              />
            </Form.Item>
          </Form>
        )}
      </Modal>
    </div>
  );
};

export default ApiKeyManagement;
