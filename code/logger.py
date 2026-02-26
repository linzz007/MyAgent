"""日志模块：记录表格问答系统每个模块的执行逻辑和关键信息。

本模块提供统一的日志接口，用于记录：
- InputHandler: 输入预处理和格式统一
- FeatureExtractor: 任务特征抽取
- DifficultyScorer: 难度评分
- Router: 路径路由决策
- SimplePathAgent: 简单路径推理
- Planner: 复杂路径规划
- Calculator: 代码执行和计算
- LightCritic: 轻量校验
- Orchestrator: 流程编排
"""

import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class TableQALogger:
    """表格问答系统专用日志记录器。
    
    提供分级日志记录功能，支持记录每个模块的执行逻辑和关键信息。
    """
    
    def __init__(self, name: str = "TableQA", level: int = logging.INFO, log_file: Optional[str] = None):
        """初始化日志记录器。
        
        Args:
            name: 日志记录器名称
            level: 日志级别（DEBUG, INFO, WARNING, ERROR）
            log_file: 日志文件路径（如果为None，则只输出到控制台）
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 控制台输出格式
            console_format = logging.Formatter(
                '[%(asctime)s] [%(name)s] [%(levelname)s] [%(module)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
            
            # 文件输出（如果指定）
            if log_file:
                file_format = logging.Formatter(
                    '[%(asctime)s] [%(name)s] [%(levelname)s] [%(module)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(file_format)
                self.logger.addHandler(file_handler)
    
    def log_input_handler(self, question: str, table_info: Dict[str, Any]) -> None:
        """记录输入处理模块的执行逻辑。
        
        Args:
            question: 用户问题
            table_info: 表格信息（行数、列数、列名等）
        """
        self.logger.info("=" * 80)
        self.logger.info("[InputHandler] 输入管理模块开始处理")
        self.logger.info(f"  - 问题: {question}")
        self.logger.info(f"  - 表格行数: {table_info.get('num_rows', 'N/A')}")
        self.logger.info(f"  - 表格列数: {table_info.get('num_cols', 'N/A')}")
        self.logger.info(f"  - 列名: {table_info.get('columns', [])}")
        self.logger.info("[InputHandler] 输入管理模块处理完成")
    
    def log_feature_extractor(self, semantic_features: Dict[str, Any], 
                              structural_features: Dict[str, Any], 
                              coarse_intent: str) -> None:
        """记录特征抽取模块的执行逻辑。
        
        Args:
            semantic_features: 语义特征字典
            structural_features: 结构特征字典
            coarse_intent: 粗粒度意图标签
        """
        self.logger.info("=" * 80)
        self.logger.info("[FeatureExtractor] 任务特征抽取模块开始执行")
        self.logger.info(f"  - 粗粒度意图: {coarse_intent}")
        self.logger.info(f"  - 语义特征: {semantic_features}")
        self.logger.info(f"  - 结构特征: {structural_features}")
        self.logger.info("[FeatureExtractor] 任务特征抽取模块执行完成")
    
    def log_difficulty_scorer(self, semantic_score: float, cell_score: float,
                              difficulty_score: float, difficulty_level: str) -> None:
        """记录难度评分模块的执行逻辑。
        
        Args:
            semantic_score: 语义难度分数
            cell_score: 结构复杂度分数
            difficulty_score: 综合难度分数
            difficulty_level: 难度等级（easy/medium/hard）
        """
        self.logger.info("=" * 80)
        self.logger.info("[DifficultyScorer] 难度评分模块开始执行")
        self.logger.info(f"  - 语义难度分数: {semantic_score:.3f}")
        self.logger.info(f"  - 结构复杂度分数: {cell_score:.3f}")
        self.logger.info(f"  - 综合难度分数: {difficulty_score:.3f}")
        self.logger.info(f"  - 难度等级: {difficulty_level}")
        self.logger.info("[DifficultyScorer] 难度评分模块执行完成")
    
    def log_router(self, route_type: str, routing_context: Dict[str, Any]) -> None:
        """记录路径路由模块的执行逻辑。
        
        Args:
            route_type: 路径类型（SIMPLE/COMPLEX）
            routing_context: 路由上下文信息
        """
        self.logger.info("=" * 80)
        self.logger.info("[Router] 路径路由模块开始执行")
        self.logger.info(f"  - 路由决策: {route_type}")
        self.logger.info(f"  - 路由上下文: {routing_context}")
        self.logger.info(f"[Router] 路径路由模块执行完成，选择路径: {route_type}")
    
    def log_simple_path(self, question: str, answer: str, reasoning_trace: str = "") -> None:
        """记录简单路径代理的执行逻辑。
        
        Args:
            question: 用户问题
            answer: 生成的答案
            reasoning_trace: 推理轨迹（可选）
        """
        self.logger.info("=" * 80)
        self.logger.info("[SimplePathAgent] 简单路径链式推理代理开始执行")
        self.logger.info(f"  - 问题: {question}")
        self.logger.info(f"  - 推理轨迹: {reasoning_trace if reasoning_trace else '（使用LLM直接推理）'}")
        self.logger.info(f"  - 最终答案: {answer}")
        self.logger.info("[SimplePathAgent] 简单路径链式推理代理执行完成")
    
    def log_planner(self, question: str, plan_steps: list, code_str: str,
                    routing_context: Dict[str, Any]) -> None:
        """记录复杂路径规划模块的执行逻辑。
        
        Args:
            question: 用户问题
            plan_steps: 规划步骤列表
            code_str: 生成的代码字符串
            routing_context: 路由上下文（包含难度等信息）
        """
        self.logger.info("=" * 80)
        self.logger.info("[Planner] 复杂路径规划代理开始执行")
        self.logger.info(f"  - 问题: {question}")
        self.logger.info(f"  - 难度等级: {routing_context.get('difficulty_level', 'N/A')}")
        self.logger.info(f"  - 规划步骤数: {len(plan_steps)}")
        for i, step in enumerate(plan_steps, 1):
            self.logger.info(f"    Step {i}: {step}")
        self.logger.info(f"  - 生成代码长度: {len(code_str)} 字符")
        self.logger.debug(f"  - 生成代码:\n{code_str}")
        self.logger.info("[Planner] 复杂路径规划代理执行完成")
    
    def log_calculator(self, code_str: str, exec_success: bool, 
                      final_value: Any, exec_error: Optional[str] = None) -> None:
        """记录计算执行模块的执行逻辑。
        
        Args:
            code_str: 执行的代码字符串
            exec_success: 执行是否成功
            final_value: 最终计算结果
            exec_error: 执行错误信息（如果有）
        """
        self.logger.info("=" * 80)
        self.logger.info("[Calculator] 工具执行与计算模块开始执行")
        self.logger.info(f"  - 执行状态: {'成功' if exec_success else '失败'}")
        if exec_success:
            self.logger.info(f"  - 最终计算结果: {final_value}")
            self.logger.info(f"  - 结果类型: {type(final_value).__name__}")
        else:
            self.logger.warning(f"  - 执行错误: {exec_error}")
        self.logger.info("[Calculator] 工具执行与计算模块执行完成")
    
    def log_critic(self, question: str, plan_steps: list, exec_success: bool,
                   final_value: Any, verdict: str, feedback: str) -> None:
        """记录轻量校验模块的执行逻辑。
        
        Args:
            question: 用户问题
            plan_steps: 规划步骤列表
            exec_success: 执行是否成功
            final_value: 最终计算结果
            verdict: 校验结果（PASS/REPLAN）
            feedback: 校验反馈信息
        """
        self.logger.info("=" * 80)
        self.logger.info("[LightCritic] 轻量校验模块开始执行")
        self.logger.info(f"  - 问题: {question}")
        self.logger.info(f"  - 执行状态: {'成功' if exec_success else '失败'}")
        if exec_success:
            self.logger.info(f"  - 计算结果: {final_value}")
        self.logger.info(f"  - 校验结果: {verdict}")
        self.logger.info(f"  - 校验反馈: {feedback}")
        self.logger.info("[LightCritic] 轻量校验模块执行完成")
    
    def log_orchestrator(self, question: str, route_type: str, 
                        final_answer: str, meta: Dict[str, Any]) -> None:
        """记录流程编排器的执行逻辑。
        
        Args:
            question: 用户问题
            route_type: 路径类型（SIMPLE/COMPLEX）
            final_answer: 最终答案
            meta: 元信息（包含执行步骤、错误等信息）
        """
        self.logger.info("=" * 80)
        self.logger.info("[Orchestrator] 流程编排器开始执行")
        self.logger.info(f"  - 问题: {question}")
        self.logger.info(f"  - 选择路径: {route_type}")
        if route_type == "COMPLEX":
            self.logger.info(f"  - 规划步骤数: {len(meta.get('plan_steps', []))}")
            self.logger.info(f"  - 执行状态: {'成功' if meta.get('exec_success', False) else '失败'}")
            self.logger.info(f"  - 校验结果: {meta.get('critic_verdict', 'N/A')}")
            self.logger.info(f"  - 重规划次数: {meta.get('replan_count', 0)}")
        self.logger.info(f"  - 最终答案: {final_answer}")
        self.logger.info("[Orchestrator] 流程编排器执行完成")
    
    def debug(self, msg: str) -> None:
        """记录调试信息。"""
        self.logger.debug(msg)
    
    def info(self, msg: str) -> None:
        """记录普通信息。"""
        self.logger.info(msg)
    
    def warning(self, msg: str) -> None:
        """记录警告信息。"""
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        """记录错误信息。"""
        self.logger.error(msg)


# 全局日志记录器实例（单例模式）
_global_logger: Optional[TableQALogger] = None


def get_logger(log_file: Optional[str] = None) -> TableQALogger:
    """获取全局日志记录器实例。
    
    Args:
        log_file: 日志文件路径（仅在第一次调用时生效）
        
    Returns:
        TableQALogger实例
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = TableQALogger(log_file=log_file)
    return _global_logger

