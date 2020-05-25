# -*- coding: utf-8 -*-
# @Time    : 2020/5/6 23:37
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : kb.py
# @Software: PyCharm

#
from neo4j import GraphDatabase
import time

# define neo4j
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', '123456'))
session = driver.session()
session.run('MATCH (n) OPTIONAL MATCH (n)-[r]->() RETURN count(n.name) + count(r)')

def GetRelationPathsBack(entity):
    """
    根据实体名称或者所有1跳关系，反向
    :param entity:
    :return:
    """
    cql_1 = "match ()-[r1:Relation]->(a:Entity{entity:$name})  return DISTINCT r1.name"
    cql_2 = "match ()-[r1:Relation]->()-[r2:Relation]->(a:Entity{entity:$name})  return DISTINCT r1.name, r2.name"
    rpaths1 = []
    res = session.run(cql_1, name=entity['entity'])
    for record in res:
        rpaths1.append([record['r1.name']])
    # rpaths2 = []
    # res = session.run(cql_2, name=entity['entity'])
    # for record in res:
    #     rpaths2.append([record['r1.name'], record['r2.name']])
    return rpaths1

def GetRelationPathsForward(entity):
    """
    根据实体名称或者所有1跳2跳3跳关系
    :param entity:
    :return:
    """
    cql_1 = "match (a:Entity{entity:$name})-[r1:Relation]->()  return DISTINCT r1.name"
    cql_2 = "match (a:Entity{entity:$name})-[r1:Relation]->()-[r2:Relation]->()  return DISTINCT r1.name, r2.name"
    cql_3 = "match (a:Entity{entity:$name})-[r1:Relation]->()-[r2:Relation]->()-[r3:Relation]->()  return DISTINCT r1.name, r2.name, r3.name"
    rpaths1 = []
    res = session.run(cql_1, name=entity['entity'])
    for record in res:
        rpaths1.append([record['r1.name']])
    rpaths2 = []
    res = session.run(cql_2, name=entity['entity'])
    for record in res:
        rpaths2.append([record['r1.name'], record['r2.name']])
    rpaths3 = []
    res = session.run(cql_3, name=entity['entity'])
    for record in res:
        rpaths3.append([record['r1.name'], record['r2.name'], record['r3.name']])
    return rpaths1 + rpaths2 + rpaths3


def GetRelationPathsSingle(entity):
    """
    根据实体名称或者所有1跳关系
    :param entity:
    :return:
    """
    cql_1 = "match (a:Entity)-[r1:Relation]-() where a.name=$name return DISTINCT r1.name"
    rpaths1 = []
    res = session.run(cql_1, name=entity)
    for record in res:
        rpaths1.append([record['r1.name']])
    return rpaths1

def GetRelationPathsTwo(entity):
    """
    根据实体名称或者所有1跳2跳关系
    :param entity:
    :return:
    """
    cql_2 = "match (a:Entity{name:$name})-[r1:Relation]-()-[r2:Relation]-()  return DISTINCT r1.name, r2.name"
    rpaths2 = []
    res = session.run(cql_2, name=entity)
    for record in res:
        rpaths2.append([record['r1.name'], record['r2.name']])
    return rpaths2

def GetRelationNum(entity):
    """
    根据实体名称得到与之相连的关系数量，代表实体在知识库种的流行度
    :param entity:
    :return:
    """
    cql = "match p=(a:Entity{name:$name})-[r1:Relation]-() return count(p)"
    res = session.run(cql, name=entity)
    ans = 0
    for record in res:
        ans = record.values()[0]
    return ans

def GetRelationNumsig(entity):
    '''根据实体名，得到一跳内的不同关系个数,重要度'''
    cql= "match p=(a:Entity{name:$name})-[r1:Relation]-() return DISTINCT r1.name"
    rpath = []
    res = session.run(cql,name=entity)
    for record in res:
        rpath.append(record['r1.name'])
    return len(set(rpath))

def GetTwoEntityTuple(e1, r1, e2):
    """
    根据两个实体，查找满足其1跳关系相连时的2跳关系
    :param e1:
    :param r1:
    :param e2:
    :return:
    """
    cql = "match (a:Entity{name:$e1n)-[r1:Relation{name:$r1n}]-(b:Entity)-[r2:Relation]-(c:Entity{name:$e2n}) return DISTINCT r2.name"
    tuples = []
    res = session.run(cql,e1n=e1, r1n=r1, e2n=e2)
    for record in res:
        tuples.append(tuple([e1,r1,record['r2.name'],e2]))
    return tuples



def GetTwoEntityRelation(e1, e2):
    """
    根据两个实体，查找满足相连时的2跳关系
    :param e1:
    :param r1:
    :param e2:
    :return:
    """
    cql = "match (a:Entity{entity:$e1n})-[r1:Relation]-(b:Entity)-[r2:Relation]-(c:Entity{entity:$e2n}) return DISTINCT r1.name, r2.name"
    tuples = []
    res = session.run(cql,e1n=e1['entity'],e2n=e2['entity'])
    for record in res:
        tuples.append((record['r1.name'], record['r2.name']))
    return tuples

def GetEntity2ThreeRelation(e1, r1, r2, r3):
    cql = "match (a:Entity{entity:$e1n})-[r1:Relation{name:$r1n}]->(b:Entity)-[r2:Relation{name:$r2n}]->(c:Entity)-[r3:Relation{name:$r3n}]->(d:Entity) return DISTINCT d.name"
    res = session.run(cql, e1n=e1, r1n=r1, r2n=r2, r3n=r3)
    return '\t'.join([i[0] for i in res.values()])

def GetEntity2TwoRelation(e1, r1, r2):
    cql = "match (a:Entity{entity:$e1n})-[r1:Relation{name:$r1n}]->(b:Entity)-[r2:Relation{name:$r2n}]->(c:Entity) return DISTINCT c.name"
    res = session.run(cql, e1n=e1, r1n=r1, r2n=r2)
    return '\t'.join([i[0] for i in res.values()])

def GetAnsFword(e1, r1):
    if len(r1) == 1:
        cql = "match (a:Entity{entity:$e1n})-[r1:Relation{name:$r1n}]->(b:Entity) return DISTINCT b.name"
        res = session.run(cql, e1n=e1['entity'], r1n=r1[0])
    elif len(r1) == 2:
        cql = "match (a:Entity{entity:$e1n})-[r1:Relation{name:$r1n}]->(b:Entity)-[r2:Relation{name:$r2n}]->(c:Entity) return DISTINCT c.name"
        res = session.run(cql, e1n=e1['entity'], r1n=r1[0], r2n=r1[1])
    elif len(r1) == 3:
        cql = "match (a:Entity{entity:$e1n})-[r1:Relation{name:$r1n}]->(b:Entity)-[r2:Relation{name:$r2n}]->(c:Entity)-[r3:Relation{name:$r3n}]->(d:Entity) return DISTINCT d.name"
        res = session.run(cql, e1n=e1['entity'], r1n=r1[0], r2n=r1[1], r3n=r1[2])
    else:
        print('relation is invalid')
        return ''
    return '\t'.join([i[0] for i in res.values()])

def GetAnsBack(e1, r1):
    if len(r1) == 1:
        cql = "match (a:Entity{entity:$e1n})<-[r1:Relation{name:$r1n}]-(b:Entity) return DISTINCT b.name"
        res = session.run(cql, e1n=e1['entity'], r1n=r1[0])
    elif len(r1) == 2:
        cql = "match (a:Entity{entity:$e1n})<-[r1:Relation{name:$r1n}]-(b:Entity)<-[r2:Relation{name:$r2n}]-(c:Entity) return DISTINCT c.name"
        res = session.run(cql, e1n=e1['entity'], r1n=r1[0], r2n=r1[1])
    else:
        print('relation is invalid')
        return ''
    return '\t'.join([i[0] for i in res.values()])

def GetEntity2Tuple(e1, r1, r2, e3):
    cql = "match (a:Entity{entity:$e1n})-[r1:Relation{name:$r1n}]-(b:Entity)-[r2:Relation{name:$r2n}]-(c:Entity{entity:$e3n}) return DISTINCT b.name"
    res = session.run(cql, e1n=e1['entity'], r1n=r1, r2n=r2, e3n=e3['entity'])
    return '\t'.join([i[0] for i in res.values()])


# from py2neo import Node, Relationship, Graph, NodeMatcher, RelationshipMatcher
#
# graph = Graph('http:localhost:7474', user_name='neo4j', password='123456')
#
# r = RelationshipMatcher(graph)
# result = r.match(nodes='魅族')
# print(result.)


import time
strat = time.time()
entity = {'name': '证监会', 'entity': 'e8540359'}
res = GetRelationPathsForward(entity)
print(time.time() - strat)
print(len(res))