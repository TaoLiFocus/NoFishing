import React, { useEffect, useState } from 'react';
import {
  Table,
  Card,
  Button,
  Space,
  Modal,
  Form,
  Input,
  Switch,
  Select,
  message,
  Popconfirm,
  Tag,
} from 'antd';
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  ReloadOutlined,
  ImportOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { blacklistApi } from '../api/blacklist';
import type { BlacklistEntry } from '../types';
import { useRequireAuth } from '../hooks/useRequireAuth';
import { useAuth } from '../hooks/useAuth';

const Blacklist: React.FC = () => {
  useRequireAuth();
  const { isAdmin } = useAuth();
  const [data, setData] = useState<BlacklistEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [importModalVisible, setImportModalVisible] = useState(false);
  const [editing, setEditing] = useState<BlacklistEntry | null>(null);
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([]);
  const [form] = Form.useForm();
  const [importForm] = Form.useForm();
  const [pagination, setPagination] = useState({ current: 1, pageSize: 10, total: 0 });

  const fetchBlacklist = async (page = 1, size = 10) => {
    setLoading(true);
    try {
      const response = await blacklistApi.getBlacklist({ page: page - 1, size });
      setData(response.data.content);
      setPagination({
        current: page,
        pageSize: size,
        total: response.data.totalElements,
      });
    } catch (error) {
      message.error('获取黑名单失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBlacklist();
  }, []);

  const handleCreate = () => {
    setEditing(null);
    form.resetFields();
    setModalVisible(true);
  };

  const handleEdit = (record: BlacklistEntry) => {
    setEditing(record);
    form.setFieldsValue(record);
    setModalVisible(true);
  };

  const handleDelete = async (id: number) => {
    try {
      await blacklistApi.deleteBlacklistEntry(id);
      message.success('删除成功');
      fetchBlacklist(pagination.current, pagination.pageSize);
    } catch (error) {
      message.error('删除失败');
    }
  };

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      if (editing) {
        await blacklistApi.updateBlacklistEntry(editing.id, values);
        message.success('更新成功');
      } else {
        await blacklistApi.createBlacklistEntry(values);
        message.success('创建成功');
      }
      setModalVisible(false);
      fetchBlacklist(pagination.current, pagination.pageSize);
    } catch (error) {
      message.error('操作失败');
    }
  };

  const handleBatchImport = async () => {
    try {
      const values = await importForm.validateFields();
      const patterns = values.patterns
        .split('\n')
        .map((p: string) => p.trim())
        .filter((p: string) => p.length > 0);

      if (patterns.length === 0) {
        message.error('请输入至少一个域名模式');
        return;
      }

      const response = await blacklistApi.batchImport({
        patterns,
        comment: values.comment,
      });

      message.success(`批量导入完成: ${response.data.total} 条`);
      setImportModalVisible(false);
      importForm.resetFields();
      fetchBlacklist(pagination.current, pagination.pageSize);
    } catch (error) {
      message.error('批量导入失败');
    }
  };

  const handleBatchDelete = async () => {
    if (selectedRowKeys.length === 0) {
      message.warning('请先选择要删除的条目');
      return;
    }

    try {
      await blacklistApi.batchDelete(selectedRowKeys as number[]);
      message.success(`成功删除 ${selectedRowKeys.length} 条`);
      setSelectedRowKeys([]);
      fetchBlacklist(pagination.current, pagination.pageSize);
    } catch (error) {
      message.error('批量删除失败');
    }
  };

  const columns: ColumnsType<BlacklistEntry> = [
    { title: 'ID', dataIndex: 'id', key: 'id', width: 80 },
    { title: '模式', dataIndex: 'pattern', key: 'pattern', ellipsis: true },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      width: 120,
    },
    {
      title: '威胁类型',
      dataIndex: 'threatType',
      key: 'threatType',
      width: 120,
      render: (type?: string) => type ? <Tag color="red">{type}</Tag> : '-',
    },
    {
      title: '启用',
      dataIndex: 'enabled',
      key: 'enabled',
      width: 100,
      render: (enabled: boolean) => (enabled ? <Tag color="green">是</Tag> : <Tag color="red">否</Tag>),
    },
    {
      title: '备注',
      dataIndex: 'comment',
      key: 'comment',
      ellipsis: true,
    },
    {
      title: '操作',
      key: 'actions',
      width: 180,
      fixed: 'right',
      render: (_, record) => (
        <Space size="small">
          <Button type="link" size="small" icon={<EditOutlined />} onClick={() => handleEdit(record)}>
            编辑
          </Button>
          <Popconfirm
            title="确定要删除吗？"
            onConfirm={() => handleDelete(record.id)}
            disabled={!isAdmin()}
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />} disabled={!isAdmin()}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const rowSelection = isAdmin()
    ? {
        selectedRowKeys,
        onChange: (newSelectedRowKeys: React.Key[]) => {
          setSelectedRowKeys(newSelectedRowKeys);
        },
      }
    : undefined;

  return (
    <div>
      <h2>黑名单管理</h2>

      <Card>
        <Space style={{ marginBottom: 16 }}>
          <Button type="primary" icon={<PlusOutlined />} onClick={handleCreate}>
            添加条目
          </Button>
          {isAdmin() && (
            <>
              <Button icon={<ImportOutlined />} onClick={() => setImportModalVisible(true)}>
                批量导入
              </Button>
              <Button
                danger
                icon={<DeleteOutlined />}
                onClick={handleBatchDelete}
                disabled={selectedRowKeys.length === 0}
              >
                批量删除 ({selectedRowKeys.length})
              </Button>
            </>
          )}
          <Button icon={<ReloadOutlined />} onClick={() => fetchBlacklist()}>
            刷新
          </Button>
        </Space>

        <Table
          columns={columns}
          dataSource={data}
          loading={loading}
          rowKey="id"
          rowSelection={rowSelection}
          pagination={{
            ...pagination,
            showSizeChanger: true,
            showTotal: (total) => `共 ${total} 条`,
            onChange: (page, pageSize) => fetchBlacklist(page, pageSize),
          }}
        />
      </Card>

      <Modal
        title={editing ? '编辑黑名单条目' : '添加黑名单条目'}
        open={modalVisible}
        onOk={handleSubmit}
        onCancel={() => setModalVisible(false)}
        width={600}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="pattern"
            label="匹配模式"
            rules={[{ required: true, message: '请输入模式' }]}
            extra="支持通配符：*.example.com、example.com/*"
          >
            <Input placeholder="*.example.com" />
          </Form.Item>

          <Form.Item name="type" label="类型">
            <Input placeholder="domain、url 等" />
          </Form.Item>

          <Form.Item name="threatType" label="威胁类型">
            <Select placeholder="选择威胁类型" allowClear>
              <Select.Option value="phishing">钓鱼</Select.Option>
              <Select.Option value="malware">恶意软件</Select.Option>
              <Select.Option value="scam">诈骗</Select.Option>
              <Select.Option value="fraud">欺诈</Select.Option>
              <Select.Option value="other">其他</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="enabled"
            label="启用"
            valuePropName="checked"
            initialValue={true}
          >
            <Switch />
          </Form.Item>

          <Form.Item name="comment" label="备注">
            <Input.TextArea rows={3} placeholder="可选备注" />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="批量导入黑名单"
        open={importModalVisible}
        onOk={handleBatchImport}
        onCancel={() => setImportModalVisible(false)}
        width={600}
      >
        <Form form={importForm} layout="vertical">
          <Form.Item
            name="patterns"
            label="域名模式列表"
            rules={[{ required: true, message: '请输入域名模式' }]}
            extra="每行一个域名模式，支持通配符。例如：*.example.com 或 example.com/*"
          >
            <Input.TextArea
              rows={10}
              placeholder="*.malicious.com&#10;*phishing*.net&#10;scam-site.org/*"
            />
          </Form.Item>

          <Form.Item name="comment" label="备注">
            <Input placeholder="批量导入" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default Blacklist;
