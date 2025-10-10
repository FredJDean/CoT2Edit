import wikipedia
import spacy

# Optional: Select only the data that is consistent with the facts.

nlp = spacy.load("en_core_web_sm")


def extract_entity_and_relation(question):
    """
    简单使用 SpaCy 解析问题并提取实体和谓词
    """
    doc = nlp(question)
    entity = None
    relation = None
    for ent in doc.ents:
        entity = ent.text
        break  # 只取第一个实体
    for token in doc:
        if token.dep_ == "ROOT":
            relation = token.text
            break
    return entity, relation


def get_wiki_summary(entity, sentences=5):
    """
    获取Wikipedia上实体的摘要
    """
    try:
        summary = wikipedia.summary(entity, sentences=sentences)
        return summary
    except Exception as e:
        return f"Error retrieving Wikipedia data: {e}"


def search_answer_in_summary(summary, relation_keywords):
    """
    在Wiki摘要中搜索匹配关系词的句子
    """
    for sentence in summary.split('. '):
        if any(keyword.lower() in sentence.lower() for keyword in relation_keywords):
            return sentence.strip()
    return "Answer not found in summary."


def wiki_based_answer(question):
    entity, relation = extract_entity_and_relation(question)
    if not entity or not relation:
        return "Could not extract entity or relation."
    print(f"[Extracted] Entity: {entity}, Relation: {relation}")

    summary = get_wiki_summary(entity)
    print(f"[Wiki Summary]\n{summary}\n")

    relation_keywords = [relation, relation + "s", relation + "ed"]

    answer = search_answer_in_summary(summary, relation_keywords)
    return answer

question = "Where was Elon Musk born?"
print("[Final Answer]:", wiki_based_answer(question))
