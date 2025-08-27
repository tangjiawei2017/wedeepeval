import os
import asyncio
import pandas as pd
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig, ContextConstructionConfig
from datetime import datetime

async def main():
    # 设置环境变量
    os.environ["OPENAI_API_KEY"] = "sk-xxx"
    os.environ["OPENAI_BASE_URL"] = "url"
    model = "gpt-3.5-turbo"
    
    print("=== 使用DeepEval Synthesizer从文档生成数据集 ===")
    
    try:
        styling_config = StylingConfig(
            input_format="询问文档中知识的中文问题"
        )
        
        # 配置上下文构建参数来控制生成的Golden数量
        context_config = ContextConstructionConfig(
            max_contexts_per_document=5,  # 每个文档最多生成5个上下文
        )
        
        # 初始化Synthesizer
        synthesizer = Synthesizer(styling_config=styling_config, model=model)

        
        # 文档路径列表
        document_paths = ["docs/武研后勤指南补充.docx"]
        
        print(f"开始从文档生成Golden: {document_paths}")
        
        # 使用DeepEval的generate_goldens_from_docs方法
        goldens = synthesizer.generate_goldens_from_docs(
            document_paths=document_paths,
            include_expected_output=True,  # 生成期望输出
            max_goldens_per_context=2,    # 每个上下文生成2个Golden
            context_construction_config=context_config
        )
        
        print(f"成功生成 {len(goldens)} 个Golden")
        
        # 显示生成的Golden
        for i, golden in enumerate(goldens[:5]):  # 只显示前5个
            print(f"\n=== Golden {i+1} ===")
            print(f"问题: {golden.input}")
            print(f"期望输出: {golden.expected_output}")
            if hasattr(golden, 'context') and golden.context:
                print(f"上下文长度: {len(str(golden.context))} 字符")
        
        # 保存到CSV文件
        test_data = []
        for golden in goldens:
            test_data.append({
                'question': golden.input,
                'expected_output': getattr(golden, 'expected_output', ''),
                'context': str(getattr(golden, 'context', ''))[:500] + '...' if hasattr(golden, 'context') and len(str(getattr(golden, 'context', ''))) > 500 else str(getattr(golden, 'context', '')),
                'context_length': len(str(getattr(golden, 'context', '')))
            })
        
        df = pd.DataFrame(test_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"deepeval_goldens_{model}_{timestamp}.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nGolden数据集已保存到: {output_file}")
        print(f"总共生成 {len(goldens)} 个Golden")
        
        # 显示统计信息
        print(f"\n=== 统计信息 ===")
        print(f"平均上下文长度: {df['context_length'].mean():.0f} 字符")
        print(f"最短上下文: {df['context_length'].min()} 字符")
        print(f"最长上下文: {df['context_length'].max()} 字符")
        
    except Exception as e:
        print(f"生成Golden时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
