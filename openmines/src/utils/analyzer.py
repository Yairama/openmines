import re
import json
import time
import os
import inspect
import argparse
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from importlib import import_module
import pkgutil

class LogAnalyzer:
    def __init__(self, api_key, api_base="https://api.siliconflow.cn", model_name="deepseek-chat", language="English"):
        self.model_name = model_name
        self.language = language
        self.time_intervals = [
            (0, 80, "0-80 minutes"),
            (80, 160, "80-160 minutes"),
            (160, 240, "160-240 minutes")
        ]
        self.summaries = []
        self.console = Console()
        
        # Initialize API client (supports both the new and legacy SDKs)
        try:
            # Attempt to use the modern OpenAI client
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=api_base)
            self.api_version = "new"
        except ImportError:
            # Fall back to the legacy openai package
            import openai
            openai.api_key = api_key
            openai.api_base = "https://api.siliconflow.cn"
            self.client = openai
            self.api_version = "old"

    def extract_time(self, log_line):
        """Extract the timestamp from a log line via regex."""
        time_match = re.search(r"Time:<([\d.]+)>", log_line)
        return float(time_match.group(1)) if time_match else None

    def categorize_logs(self, log_path):
        """Group log entries into predefined time intervals."""
        categorized = {interval[2]: [] for interval in self.time_intervals}
        
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                log_time = self.extract_time(line)
                if log_time is not None:
                    for start, end, name in self.time_intervals:
                        if start <= log_time < end:
                            categorized[name].append(line.strip())
                            break
        return categorized

    def get_summary(self, prompt, max_retries=3):
        """Call API to get summary"""
        for _ in range(max_retries):
            try:
                if self.api_version == "new":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        stream=True  # Enable streaming output
                    )
                    # Handle streamed responses
                    collected_content = ""
                    self.console.print("[cyan]Generating analysis...[/cyan]")
                    for chunk in response:
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                            content = chunk.choices[0].delta.content
                            if content:
                                collected_content += content
                                # Print incremental content for real-time feedback
                                self.console.print(content, end="")
                    self.console.print("\n")
                    return collected_content
                else:
                    # Streaming response handling for the legacy API
                    response = self.client.ChatCompletion.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        stream=True  # Enable streaming output
                    )
                    # Handle streamed responses
                    collected_content = ""
                    self.console.print("[cyan]Generating analysis...[/cyan]")
                    for chunk in response:
                        if 'choices' in chunk and len(chunk.choices) > 0:
                            if 'delta' in chunk.choices[0] and 'content' in chunk.choices[0].delta:
                                content = chunk.choices[0].delta.content
                                if content:
                                    collected_content += content
                                    # Print incremental content for real-time feedback
                                    self.console.print(content, end="")
                    self.console.print("\n")
                    return collected_content
                
            except Exception as e:
                self.console.print(f"[bold red]API call failed, retrying... Error: {str(e)}[/bold red]")
                time.sleep(2)
        return "Analysis failed"

    def get_dispatcher_code(self, algo_name):
        """Retrieve dispatcher source code by scanning the package."""
        try:
            # Walk every module registered under dispatch_algorithms
            dispatchers_package = 'openmines.src.dispatch_algorithms'
            package = import_module(dispatchers_package)
            
            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                module = import_module(f"{dispatchers_package}.{module_name}")
                # Look for the matching class definition
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and name == algo_name:
                        return inspect.getsource(obj)
            return f"未找到{algo_name}类的实现"
        except Exception as e:
            return f"获取调度算法代码失败: {str(e)}"

    def analyze_logs(self, log_path):
        """Main analysis function"""
        self.console.print("[bold blue]Starting log file analysis...[/bold blue]")
        import pathlib
        
        # Normalize the log path and locate the newest log file if a directory is provided
        path = pathlib.Path(log_path)
        if path.is_dir():
            log_files = list(path.glob("*.log"))
            if not log_files:
                self.console.print(f"[bold red]Error: No log files found in directory {log_path}[/bold red]")
                return ""
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            log_path = str(latest_log)
            self.console.print(f"[bold blue]Analyzing latest log file: {latest_log.name}[/bold blue]")
        
        dispatcher_sections = self.identify_dispatcher_sections(log_path)
        
        # Display dispatcher sections discovered in the log
        self.console.print("\n[bold cyan]Identified dispatcher algorithm sections:[/bold cyan]")
        for section in dispatcher_sections:
            self.console.print(f"• {section['name']}: Lines {section['start_line']}-{section['end_line']}")

        # Determine which sections should be analyzed
        if hasattr(self, 'dispatcher_name') and self.dispatcher_name:
            selected_sections = [s for s in dispatcher_sections if s['name'] == self.dispatcher_name]
            if not selected_sections:
                self.console.print(f"[bold red]Error: Specified dispatcher {self.dispatcher_name} not found[/bold red]")
                return ""
            self.console.print(f"\n[bold green]Analyzing specified algorithm: {self.dispatcher_name}[/bold green]")
        else:
            if len(dispatcher_sections) > 1:
                latest_section = max(dispatcher_sections, key=lambda x: x['end_line'])
                selected_sections = [latest_section]
                self.console.print(f"[bold yellow]⚠ Multiple dispatcher sections detected, automatically selecting the latest: {latest_section['name']}[/bold yellow]")
            else:
                selected_sections = dispatcher_sections

        # Analyze each chosen section sequentially
        final_report = ""
        for section in selected_sections:
            self.console.print(f"\n[bold magenta]Analyzing {section['name']} section...[/bold magenta]")
            report = self.analyze_section(log_path, section)
            final_report += f"## {section['name']} Analysis Report\n\n{report}\n\n"
        
        return final_report

    def identify_dispatcher_sections(self, log_path):
        """Identify dispatcher-specific sections within the log file."""
        sections = []
        current_section = None
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Detect the start of a dispatcher section
                start_match = re.search(r'simulation started with dispatcher (\w+)', line)
                if start_match:
                    if current_section:  # Close a section that did not end cleanly
                        current_section['end_line'] = line_num - 1
                        sections.append(current_section)
                    current_section = {
                        'name': start_match.group(1),
                        'start_line': line_num,
                        'end_line': None
                    }
                
                # Detect the end of the dispatcher section
                end_match = re.search(r'simulation finished with dispatcher (\w+)', line)
                if end_match and current_section:
                    if end_match.group(1) == current_section['name']:
                        current_section['end_line'] = line_num
                        sections.append(current_section)
                        current_section = None
        
        return sections

    def analyze_section(self, log_path, section):
        """Analyze a single dispatcher algorithm section"""
        # Extract log lines that belong to the requested section
        section_logs = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if section['start_line'] <= line_num <= section['end_line']:
                    section_logs.append(line.strip())
        
        # Persist a temporary log file for downstream analysis
        temp_log = f"temp_{section['name']}.log"
        try:
            with open(temp_log, 'w', encoding='utf-8') as f:
                f.write("\n".join(section_logs))
            
            # Reuse the existing single-log analysis routine
            self.console.print(f"\n[bold magenta]Analyzing {section['name']} section ({len(section_logs)} lines)...[/bold magenta]")
            return self._analyze_single_log(temp_log, section['name'])
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_log):
                os.remove(temp_log)

    def _analyze_single_log(self, log_path, algo_name):
        """Wrap original analysis logic"""
        # Pull the dispatcher implementation for reference
        dispatcher_code = self.get_dispatcher_code(algo_name)
        
        # Generate summaries for each configured time window
        categorized = self.categorize_logs(log_path)
        
        self.summaries = []  # Reset any previously cached summaries
        
        for interval_name, logs in categorized.items():
            if not logs:
                continue
            
            self.console.print(f"\n[bold cyan]Analyzing {interval_name} time period...[/bold cyan]")
            
            # Construct the analysis prompt for this interval
            prompt = f"""
            You are now a mining truck dispatch analysis expert. You will analyze log data from a professional perspective and use data-rich language to summarize it, helping dispatchers improve their strategies.
            Dispatchers can only write necessary code and cannot add hardware facilities, so your suggestions should focus on dispatch strategies.
            You should focus on analyzing the current situation and problems based on data, rather than providing solutions and suggestions.

            Current dispatch algorithm code:
            ```python
            {dispatcher_code}
            ```

            Please analyze the following mining truck dispatch log segment (time range: {interval_name}) and include:
                1. Truck destinations during this period
                2. Load and unload point status
                3. Number and distribution of abnormal events (traffic jams, equipment failures)
                4. Traffic conditions
                5. Factors that may affect efficiency in the system

            Sample log format:
            [Truck: Time:<time> Truck:<name> Start moving to <destination>, distance: <number>km, speed: <number>]

            Please return the analysis report in {self.language} with concise text that directly addresses the current situation with rich data:"""

            # Append a sample of log lines (trimmed for brevity)
            prompt += "\n\nRelevant log segment:\n" + "\n".join(logs[:20])  # Limit to the first 20 entries
            
            # Retrieve and store the interval summary
            summary = self.get_summary(prompt)
            self.summaries.append({
                "interval": interval_name,
                "summary": summary
            })
            self.console.print(f"[bold green]✓ Completed analysis for {interval_name}[/bold green]")
            time.sleep(1)  # Avoid hitting API rate limits

        # Stage 2: produce an overall consolidated summary
        self.console.print("\n[bold magenta]Generating final comprehensive report...[/bold magenta]")
        
        combined_summary = "\n\n".join(
            [f"## {s['interval']}\n{s['summary']}" for s in self.summaries]
        )
        
        final_prompt = f"""
        You are now a mining truck dispatch analysis expert. You will analyze log data from a professional perspective and use data-rich language to summarize it, helping dispatchers improve their strategies.
        Dispatchers can only write necessary code and cannot add hardware facilities, so your suggestions should focus on dispatch strategies.
            Based on the following time-period summaries, please provide a comprehensive analysis of the entire mining operation:
                    1. Overall effectiveness of the dispatch strategy
                    2. System bottlenecks and potential risks
                    3. Strategy optimization suggestions
                    4. Key data metrics trend analysis

            {combined_summary}
            Strategy code:
            ```python
            {dispatcher_code}
            ```
            Please return the analysis report in {self.language} with concise text that directly addresses the current situation with rich data:"""
        
        final_report = self.get_summary(final_prompt)
        return final_report

if __name__ == "__main__":
    console = Console()
    
    # Configure command-line arguments
    parser = argparse.ArgumentParser(description='矿山卡车调度日志分析工具')
    parser.add_argument('log_path', help='日志文件路径')
    parser.add_argument('--api-key', default="sk-whknzqhqufnsnfrjtrofqmlyuxhaobawmdtfhpuvyctaoblr",
                        help='API密钥')
    parser.add_argument('--model', default="deepseek-ai/DeepSeek-V3",
                        help='模型名称')
    
    args = parser.parse_args()
    
    try:
        analyzer = LogAnalyzer(
            api_key=args.api_key,
            model_name=args.model
        )
        
        console.print("[bold blue]开始分析矿山卡车调度日志...[/bold blue]")
        
        analysis_result = analyzer.analyze_logs(args.log_path)
        
        # Persist the analysis report
        output_file = "log_analysis_report.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# 矿山卡车调度分析报告\n\n")
            f.write(analysis_result)
        
        console.print(f"[bold green]✓ 分析完成，结果已保存至{output_file}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]错误: {str(e)}[/bold red]")