"""
ExamHelperServer — BYOA 实验 MCP 服务。

基于官方 MCP Python SDK 的 FastMCP 框架，提供考试大纲解析与学术概念检索能力。
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Final
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, field_validator

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Pydantic 模型：考试大纲结构化提取
# ---------------------------------------------------------------------------


class ExamPointCategory(str, Enum):
    """考试大纲条目的分类枚举。"""

    CORE_CONCEPT = "核心概念"
    CORE_SKILL = "核心技能"
    PRACTICE = "实践要求"
    OTHER = "其他考点"


class SyllabusExamPoint(BaseModel):
    """单条考试考点的结构化表示。"""

    index: int = Field(..., ge=1, description="考点在原文中的序号")
    category: ExamPointCategory = Field(..., description="考点所属分类")
    title: str = Field(..., min_length=1, description="考点主题（清洗后的简短标题）")
    detail: str = Field(..., min_length=1, description="考点详细说明（清洗后的正文）")
    keywords: list[str] = Field(
        default_factory=list,
        description="从正文中提取的关键术语列表",
    )

    @field_validator("title", "detail", mode="before")
    @classmethod
    def strip_whitespace(cls, value: object) -> object:
        """去除首尾空白并压缩连续空格。"""
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value.strip())
        return value


class SyllabusSummary(BaseModel):
    """整份考试大纲的结构化摘要。"""

    source_file: str = Field(..., description="源文件路径")
    course_title: str = Field(default="", description="课程/考试名称")
    exam_points: list[SyllabusExamPoint] = Field(
        default_factory=list,
        description="按序号排列的核心考点列表",
    )
    total_points: int = Field(default=0, ge=0, description="考点总数")

    def to_readable_summary(self) -> str:
        """将结构化摘要渲染为面向大模型的可读文本。"""
        lines: list[str] = [
            "【考试大纲摘要】",
            f"来源文件: {self.source_file}",
        ]
        if self.course_title:
            lines.append(f"课程/考试: {self.course_title}")
        lines.append(f"核心考点数量: {self.total_points}")
        lines.append("")
        for point in self.exam_points:
            lines.append(f"{point.index}. [{point.category.value}] {point.title}")
            lines.append(f"   说明: {point.detail}")
            if point.keywords:
                lines.append(f"   关键词: {', '.join(point.keywords)}")
            lines.append("")
        return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# 本地备用学术字典（网络不可达时的防御式兜底）
# ---------------------------------------------------------------------------

LOCAL_ACADEMIC_DICTIONARY: Final[dict[str, str]] = {
    "acid": (
        "ACID 是关系型数据库事务的四个基本特性缩写："
        "原子性（Atomicity）——事务中的所有操作要么全部成功提交，要么全部回滚，"
        "不存在部分执行的中间状态；"
        "一致性（Consistency）——事务执行前后，数据库必须满足所有完整性约束，"
        "数据从一个合法状态转换到另一个合法状态；"
        "隔离性（Isolation）——并发执行的事务之间相互隔离，"
        "一个事务的中间状态对其他事务不可见，如同串行执行；"
        "持久性（Durability）——事务一旦提交，其对数据库的修改将永久保存，"
        "即使系统发生故障也不会丢失。"
    ),
    "atomicity": (
        "原子性（Atomicity）是 ACID 特性之一，指事务是一个不可分割的工作单元，"
        "事务中的全部数据库操作要么全部完成，要么在遇到错误时全部撤销，"
        "不允许出现部分提交的情况。"
    ),
    "consistency": (
        "一致性（Consistency）是 ACID 特性之一，指事务执行必须使数据库从一个"
        "满足所有完整性约束的合法状态转移到另一个合法状态，"
        "不会破坏实体完整性、参照完整性或用户定义的约束。"
    ),
    "isolation": (
        "隔离性（Isolation）是 ACID 特性之一，指多个并发事务同时执行时，"
        "每个事务都感觉不到其他事务的存在，就好像系统中只有它一个事务在运行。"
        "数据库通过锁机制和多版本并发控制（MVCC）等手段实现不同级别的隔离。"
    ),
    "durability": (
        "持久性（Durability）是 ACID 特性之一，指事务一旦提交成功，"
        "其对数据库的所有修改就必须永久保存，即使随后发生系统崩溃、"
        "断电等故障，已提交的数据也不会丢失。通常通过预写式日志（WAL）实现。"
    ),
    "opengauss": (
        "openGauss 是华为主导开源的企业级关系型数据库管理系统（RDBMS），"
        "基于 PostgreSQL 内核深度优化，面向企业关键业务场景。"
        "它支持 SQL 标准、ACID 事务、高可用集群、主备复制、"
        "逻辑/物理备份与恢复等企业级特性，"
        "广泛应用于金融、电信、政务等对数据安全与性能要求较高的领域。"
    ),
    "join": (
        "JOIN（连接）是 SQL 中用于根据两个或多个表之间的关联条件，"
        "将它们的行组合成结果集的操作。"
        "常见类型包括 INNER JOIN（内连接，仅返回匹配行）、"
        "LEFT JOIN（左外连接，保留左表全部行）、"
        "RIGHT JOIN（右外连接，保留右表全部行）和 FULL JOIN（全外连接）。"
    ),
    "group by": (
        "GROUP BY 是 SQL 的聚合子句，用于将结果集按指定列的值分组，"
        "通常与 COUNT、SUM、AVG、MAX、MIN 等聚合函数配合使用，"
        "对每个分组分别计算统计值。"
    ),
    "having": (
        "HAVING 是 SQL 中用于对 GROUP BY 分组后的结果进行筛选的子句，"
        "功能类似 WHERE，但 WHERE 在分组前过滤行，"
        "HAVING 在分组后过滤组，且可以使用聚合函数作为条件。"
    ),
    "transaction": (
        "数据库事务（Transaction）是作为单个逻辑工作单元执行的一系列数据库操作，"
        "具有 ACID 特性。事务通过 BEGIN/COMMIT/ROLLBACK 等语句控制，"
        "是保证数据一致性和并发正确性的基本机制。"
    ),
    "sql": (
        "SQL（Structured Query Language，结构化查询语言）是用于管理关系型数据库的"
        "国际标准语言，包括数据定义（DDL）、数据操纵（DML）、"
        "数据控制（DCL）和数据查询（DQL）等子语言。"
    ),
    "backup": (
        "数据库备份（Backup）是将数据库当前状态复制并保存到外部存储的过程，"
        "以便在数据丢失或损坏时进行恢复。"
        "常见方式包括完全备份、增量备份、逻辑备份（如 pg_dump）和物理备份。"
    ),
    "recovery": (
        "数据库恢复（Recovery）是在系统故障或人为错误导致数据损坏后，"
        "利用备份文件和事务日志将数据库还原到一致状态的过程。"
        "恢复策略通常结合全量备份与增量/日志重做（Redo）机制。"
    ),
}

# 分类前缀 → 枚举映射
_CATEGORY_PREFIX_MAP: Final[dict[str, ExamPointCategory]] = {
    "核心概念": ExamPointCategory.CORE_CONCEPT,
    "核心技能": ExamPointCategory.CORE_SKILL,
    "实践要求": ExamPointCategory.PRACTICE,
}

# 关键词提取正则：英文术语、中文括号内术语
_KEYWORD_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z][A-Za-z0-9_]*(?:\s+[A-Za-z]+)*|（[^）]+）"
)

# HTTP 请求超时（秒）
_HTTP_TIMEOUT: Final[int] = 10

# ---------------------------------------------------------------------------
# FastMCP 服务实例
# ---------------------------------------------------------------------------

mcp: FastMCP = FastMCP(
    name="ExamHelperServer",
    instructions=(
        "ExamHelperServer 提供考试大纲解析（analyze_local_syllabus）"
        "与学术概念权威检索（search_academic_concept）两项能力，"
        "适用于数据库系统原理等课程的考前复习与概念查证。"
    ),
)


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _extract_keywords(text: str) -> list[str]:
    """从考点正文中提取关键术语。"""
    raw_matches: list[str] = _KEYWORD_PATTERN.findall(text)
    keywords: list[str] = []
    seen: set[str] = set()
    for match in raw_matches:
        cleaned: str = match.strip("（）() ")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            keywords.append(cleaned)
    return keywords


def _parse_syllabus_content(raw_text: str, source_file: str) -> SyllabusSummary:
    """
    解析原始大纲文本，返回 Pydantic 结构化摘要。

    Args:
        raw_text: 从文件流读取的原始文本内容。
        source_file: 源文件路径，用于溯源。

    Returns:
        经过清洗与结构化后的 SyllabusSummary 对象。
    """
    lines: list[str] = [line.strip() for line in raw_text.splitlines() if line.strip()]
    full_text: str = "\n".join(lines)

    course_title: str = ""
    header_match: re.Match[str] | None = re.search(
        r"^(.+?[》】\)]?)[：:]\s*$", lines[0] if lines else ""
    )
    if header_match:
        course_title = header_match.group(1).strip()
    elif lines:
        first_colon: int = lines[0].find("：")
        if first_colon == -1:
            first_colon = lines[0].find(":")
        if first_colon > 0:
            course_title = lines[0][:first_colon].strip()

    exam_points: list[SyllabusExamPoint] = []
    point_pattern: re.Pattern[str] = re.compile(
        r"(\d+)\.\s*(?:\[?([^：:\]]+)[\]：:]?\s*)?(.+?)(?=\d+\.\s|$)",
        re.DOTALL,
    )

    for match in point_pattern.finditer(full_text):
        index: int = int(match.group(1))
        category_label: str = (match.group(2) or "其他考点").strip()
        body: str = match.group(3).strip().rstrip("。") + "。"

        category: ExamPointCategory = _CATEGORY_PREFIX_MAP.get(
            category_label, ExamPointCategory.OTHER
        )

        title_part: str
        detail_part: str
        colon_pos: int = body.find("：")
        if colon_pos == -1:
            colon_pos = body.find(":")
        if colon_pos > 0:
            title_part = body[:colon_pos].strip()
            detail_part = body[colon_pos + 1 :].strip()
        else:
            sentence_end: int = body.find("。")
            if 0 < sentence_end < len(body) - 1:
                title_part = body[:sentence_end].strip()
                detail_part = body[sentence_end + 1 :].strip()
            else:
                title_part = body
                detail_part = body

        exam_points.append(
            SyllabusExamPoint(
                index=index,
                category=category,
                title=title_part,
                detail=detail_part,
                keywords=_extract_keywords(body),
            )
        )

    return SyllabusSummary(
        source_file=source_file,
        course_title=course_title,
        exam_points=exam_points,
        total_points=len(exam_points),
    )


def _lookup_local_dictionary(concept_name: str) -> str | None:
    """
    在本地备用学术字典中查找概念定义。

    Args:
        concept_name: 待查询的概念名称。

    Returns:
        匹配到的定义文本；未命中时返回 None。
    """
    normalized: str = concept_name.strip().lower()
    if normalized in LOCAL_ACADEMIC_DICTIONARY:
        return LOCAL_ACADEMIC_DICTIONARY[normalized]

    for key, definition in LOCAL_ACADEMIC_DICTIONARY.items():
        if key in normalized or normalized in key:
            return definition

    return None


def _fetch_definition_from_wikipedia(concept_name: str) -> str | None:
    """
    从中文维基百科抓取概念的权威定义摘要。

    Args:
        concept_name: 待检索的学术概念名称。

    Returns:
        提取到的定义段落；抓取失败时返回 None。
    """
    headers: dict[str, str] = {
        "User-Agent": (
            "ExamHelperServer/1.0 (BYOA Academic Research; "
            "contact: exam-helper@example.edu)"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    search_url: str = (
        "https://zh.wikipedia.org/w/api.php"
        f"?action=opensearch&search={quote(concept_name)}&limit=1&namespace=0&format=json"
    )

    try:
        search_resp: requests.Response = requests.get(
            search_url, headers=headers, timeout=_HTTP_TIMEOUT
        )
        search_resp.raise_for_status()
        search_data: list[object] = search_resp.json()
        if len(search_data) < 4:
            return None

        titles: list[str] = search_data[1]  # type: ignore[assignment]
        descriptions: list[str] = search_data[2]  # type: ignore[assignment]

        if not titles:
            return None

        title: str = titles[0]
        short_desc: str = descriptions[0] if descriptions else ""

        page_url: str = f"https://zh.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        page_resp: requests.Response = requests.get(
            page_url, headers=headers, timeout=_HTTP_TIMEOUT
        )
        page_resp.raise_for_status()

        soup: BeautifulSoup = BeautifulSoup(page_resp.text, "html.parser")
        content_div = soup.find("div", {"class": "mw-parser-output"})
        if content_div is None:
            return short_desc or None

        paragraphs: list[str] = []
        for element in content_div.find_all("p", recursive=False):
            text: str = element.get_text(strip=True)
            if len(text) > 20:
                paragraphs.append(text)
            if len(paragraphs) >= 2:
                break

        body: str = "\n".join(paragraphs) if paragraphs else short_desc
        if body:
            return f"【来源: 中文维基百科 — {title}】\n{body}"
        return None

    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None


def _fetch_definition_from_baidu_baike(concept_name: str) -> str | None:
    """
    从百度百科抓取概念的摘要定义（教学镜像备选源）。

    Args:
        concept_name: 待检索的学术概念名称。

    Returns:
        提取到的定义摘要；抓取失败时返回 None。
    """
    headers: dict[str, str] = {
        "User-Agent": (
            "ExamHelperServer/1.0 (BYOA Academic Research; "
            "contact: exam-helper@example.edu)"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    url: str = f"https://baike.baidu.com/item/{quote(concept_name)}"

    try:
        resp: requests.Response = requests.get(url, headers=headers, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        soup: BeautifulSoup = BeautifulSoup(resp.text, "html.parser")

        lemma_summary = soup.find("div", class_="lemmaSummary")
        if lemma_summary is None:
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                content: str = str(meta_desc["content"]).strip()
                if content:
                    return f"【来源: 百度百科 — {concept_name}】\n{content}"
            return None

        summary_text: str = lemma_summary.get_text(separator="\n", strip=True)
        if summary_text:
            return f"【来源: 百度百科 — {concept_name}】\n{summary_text}"
        return None

    except (requests.RequestException, ValueError, KeyError):
        return None


def _search_concept_online(concept_name: str) -> str | None:
    """
    依次尝试多个权威来源检索学术概念定义。

    Args:
        concept_name: 待检索的概念名称。

    Returns:
        首个成功获取的定义文本；全部失败时返回 None。
    """
    for fetcher in (_fetch_definition_from_wikipedia, _fetch_definition_from_baidu_baike):
        result: str | None = fetcher(concept_name)
        if result:
            return result
    return None


# ---------------------------------------------------------------------------
# MCP 工具（Tools）
# ---------------------------------------------------------------------------


@mcp.tool()
def analyze_local_syllabus(file_path: str) -> str:
    """
    读取本地考试大纲文本文件，结构化提取核心考点并返回摘要。

    本工具使用标准文件流（UTF-8 编码）读取指定路径的大纲文件，
    通过 Pydantic 模型对原始文本进行解析、分类与清洗，提取每条考点的
    序号、类别（核心概念/核心技能/实践要求）、标题、详细说明及关键词，
    最终生成面向大模型消费的可读大纲摘要。

    Args:
        file_path: 本地考试大纲文本文件的路径。
            支持相对路径与绝对路径，例如 ``database_syllabus.txt``。

    Returns:
        结构化清洗后的考试大纲摘要字符串，包含课程名称、考点数量及
        逐条考点详情；若文件不存在或读取失败，返回错误说明。

    Raises:
        无显式异常抛出；所有 I/O 错误均捕获并以字符串形式返回。
    """
    resolved_path: Path = Path(file_path).expanduser().resolve()

    if not resolved_path.is_file():
        return (
            f"错误: 文件不存在或不可读 — {resolved_path}\n"
            f"请确认路径正确，例如项目根目录下的 database_syllabus.txt。"
        )

    try:
        with resolved_path.open(mode="r", encoding="utf-8") as file_stream:
            raw_content: str = file_stream.read()
    except OSError as exc:
        return f"错误: 无法读取文件 {resolved_path} — {exc}"

    if not raw_content.strip():
        return f"警告: 文件 {resolved_path} 为空，无法提取考点。"

    summary: SyllabusSummary = _parse_syllabus_content(
        raw_text=raw_content,
        source_file=str(resolved_path),
    )

    if summary.total_points == 0:
        return (
            f"警告: 未能从 {resolved_path} 中识别出编号考点。\n"
            f"原始内容预览:\n{raw_content[:500]}"
        )

    return summary.to_readable_summary()


@mcp.tool()
def search_academic_concept(concept_name: str) -> str:
    """
    检索指定学术/专业概念的权威定义，防止大模型产生概念性幻觉。

    本工具优先通过 ``requests`` 联网访问中文维基百科与百度百科等
    权威教学镜像站点，结合 ``beautifulsoup4`` 解析 HTML 并提取
    科学、准确的定义段落。当网络不可达或在线检索未命中时，
    自动回退至内置的本地备用学术字典，确保始终返回可靠定义。

    Args:
        concept_name: 待查询的专业学术概念名称。
            示例: ``ACID``、``openGauss``、``JOIN``、``事务`` 等。

    Returns:
        该概念的权威定义文本，包含来源标注；
        若在线与本地均未命中，返回提示信息及可用本地词条列表。

    Raises:
        无显式异常抛出；网络与解析异常均静默处理并触发本地兜底。
    """
    query: str = concept_name.strip()
    if not query:
        return "错误: concept_name 不能为空，请提供有效的学术概念名称。"

    online_result: str | None = _search_concept_online(query)
    if online_result:
        return online_result

    local_result: str | None = _lookup_local_dictionary(query)
    if local_result:
        return (
            f"【来源: 本地备用学术字典 — {query}】\n"
            f"（在线检索不可用或未命中，已启用防御式兜底）\n\n"
            f"{local_result}"
        )

    available_terms: str = ", ".join(sorted(LOCAL_ACADEMIC_DICTIONARY.keys()))
    return (
        f"未找到概念「{query}」的权威定义。\n"
        f"在线检索（维基百科、百度百科）均未命中，"
        f"本地备用字典中亦无完全匹配条目。\n"
        f"本地字典当前覆盖词条: {available_terms}"
    )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
