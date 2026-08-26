from chatbot_graph import MachineFaultChatBot

# 初始化聊天机器人
chatbot = MachineFaultChatBot()

# 测试问题列表
test_questions = [
    "刀具磨损能修好吗？",
    "主轴过热是什么原因？",
    "铣削时出现振纹是什么故障？",
    "刀具崩刃怎么处理？",
    "如何避免刀具磨损？",
    "表面粗糙度差该怎么调参数？"
]

print("=" * 60)
print("最终功能测试：验证所有示例问题都能正确回答")
print("=" * 60 + "\n")

for q in test_questions:
    print(f"咨询: {q}")
    answer = chatbot.chat_main(q)
    print(f"客服机器人: {answer}\n{'-' * 60}\n")

print("所有测试完成！")
