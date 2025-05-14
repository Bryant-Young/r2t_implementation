from pyspark.sql import SparkSession
import numpy as np
from pulp import *
from collections import defaultdict

import os
os.environ['HADOOP_CONF_DIR'] = "/export/server/hadoop/etc/hadoop"

queries = {
#         "Q3": """
#             select
#     c_custkey as c_id,
#     s_suppkey as s_id,
#     l_extendedprice * (1 - l_discount) as contribution
# from
#     customer,
#     orders,
#     supplier,
#     lineitem
# where
#     c_mktsegment = 'BUILDING'
#     and c_custkey = o_custkey
#     and l_orderkey = o_orderkey
#     and s_suppkey = l_suppkey
#     and o_orderdate < date '1994-03-15'
#     and l_shipdate > date '1994-03-15'
#         """,

        "Q21": """
SELECT
    supplier.s_suppkey AS s_id,
    CASE
        WHEN EXISTS (
            SELECT *
            FROM lineitem l2
            WHERE l2.l_orderkey = l1.l_orderkey
              AND l2.l_suppkey <> l1.l_suppkey
        )
        AND NOT EXISTS (
            SELECT *
            FROM lineitem l3
            WHERE l3.l_orderkey = l1.l_orderkey
              AND l3.l_suppkey <> l1.l_suppkey
              AND l3.l_receiptdate > l3.l_commitdate
        )
        THEN 1
        ELSE 0
    END AS contribution
FROM
    supplier,
    lineitem l1,
    orders,
    nation
WHERE
    supplier.s_suppkey = l1.l_suppkey
    AND orders.o_orderkey = l1.l_orderkey
    AND orders.o_orderstatus = 'F'
    AND l1.l_receiptdate > l1.l_commitdate
    AND supplier.s_nationkey = nation.n_nationkey
    AND nation.n_name = 'SAUDI ARABIA'""",

#         "Q11": """
#     WITH part_total AS ( select
#     ps_partkey,
#     sum(ps_supplycost * ps_availqty) as value
# from
#     partsupp,
#     supplier,
#     nation
# where
#     ps_suppkey = s_suppkey
#     and s_nationkey = n_nationkey
#     and n_name = 'GERMANY'
# group by
#     ps_partkey having
#        sum(ps_supplycost * ps_availqty) > (
#           select
#              sum(ps_supplycost * ps_availqty) * 0.0001
#           from
#              partsupp,
#              supplier,
#              nation
#           where
#              ps_suppkey = s_suppkey
#              and s_nationkey = n_nationkey
#              and n_name = 'GERMANY'
#        )
# order by
#     value desc )
#     SELECT
#     ps.ps_suppkey AS s_id,
#     ps.ps_supplycost * ps.ps_availqty AS contribution
# FROM
#     partsupp ps
# JOIN
#     supplier s ON ps.ps_suppkey = s.s_suppkey
# JOIN
#     nation n ON s.s_nationkey = n.n_nationkey
# JOIN
#     part_total pt ON ps.ps_partkey = pt.ps_partkey
# WHERE
#     n.n_name = 'GERMANY'
#     """,
#
#         "Q18": """select
#     c_custkey as c_id,
#     l_quantity as contribution
# from
#     customer,
#     orders,
#     lineitem
# where
#     o_orderkey in (
#        select
#           l_orderkey
#        from
#           lineitem
#        group by
#           l_orderkey having
#              sum(l_quantity) > 250
#     )
#     and c_custkey = o_custkey
#     and o_orderkey = l_orderkey
# """,
#
#         "Q10": """select
#     c_custkey as c_id,
#     l_extendedprice * (1 - l_discount) as contribution
# from
#     customer,
#     orders,
#     lineitem,
#     nation
# where
#     c_custkey = o_custkey
#     and l_orderkey = o_orderkey
#     and o_orderdate >= date '1993-10-01'
#     and o_orderdate < date '1993-10-01' + interval '3' month
#     and l_returnflag = 'R'
#     and c_nationkey = n_nationkey
# """

    }


class DPExperiment:
    def __init__(self, epsilon, GSQ, beta, primary_keys):
        self.epsilon = epsilon
        self.GSQ = GSQ
        self.beta = beta
        self.primary_keys = primary_keys
        self.spark = spark = SparkSession.builder.\
        appName("R2T_test").\
        master("yarn"). \
        config("spark.executor.memory", "4g"). \
        config("spark.executor.cores", "4"). \
        config("spark.sql.shuffle.partitions",24).\
        config("spark.sql.warehouse.dir","hdfs://node1:8020/user/hive/warehouse").\
        config("hive.metastore.uris","thrift://node1:9083").\
        enableHiveSupport().\
        getOrCreate()
        # config("spark.executor.instances", "3").\
        spark.sql("""USE ip_test""")

    def _parse_contributions(self, df):
        key_constraint_map = defaultdict(list)
        variables = {}
        true_total = 0.0
        df_results = df.collect()
        for row in df_results:
            contrib = float(row['contribution'])
            var_id = f"var_{len(variables)}"

            for pk in self.primary_keys:
                if pk in row and row[pk] is not None:
                    key = f"{pk}_{row[pk]}"
                    key_constraint_map[key].append(var_id)

            variables[var_id] = {
                'var': LpVariable(var_id, lowBound=0, upBound=contrib),
                'contrib': contrib
            }
            true_total += contrib

        return variables, key_constraint_map, true_total

    def _build_lp_problem(self, variables, key_constraint_map, tau):
        prob = LpProblem("R2T_Optimization", LpMaximize)

        for key, var_list in key_constraint_map.items():
            prob += sum(variables[var_id]['var'] for var_id in var_list) <= tau

        prob += sum(v['var'] for v in variables.values())
        return prob

    def _solve_lp(self, prob):
        try:
            prob.solve()
            if LpStatus[prob.status] == 'Optimal':
                return sum(var.value() for var in prob.variables())
            return 0.0
        except Exception as e:
            print(f"failed to solve LP: {str(e)}")
            return 0.0

    def r2t_mechanism(self, query):

        df = self.spark.sql(query)

        variables, key_constraint_map, true_total = self._parse_contributions(df)

        max_tau = self.GSQ
        tau_values = [2 ** j for j in range(int(np.log2(max_tau)) + 1)]
        print(f"\n[τ]{tau_values[:5]}... (total: {len(tau_values)})")
        noisy_results = []
        for tau in tau_values:

            prob = self._build_lp_problem(variables, key_constraint_map, tau)
            raw_value = self._solve_lp(prob)

            noise_scale = (np.log2(self.GSQ) * tau) / self.epsilon
            noise = np.random.laplace(scale=noise_scale)

            log_term = np.log(np.log2(self.GSQ) / self.beta)
            adjusted_value = (raw_value + noise) - (noise_scale * log_term)
            noisy_results.append(adjusted_value)
            print("raw_value: ", raw_value)
            print("adjusted_value: ", adjusted_value)
            print("true_total: ", true_total)

        final_result = max(noisy_results + [0])

        return final_result, true_total


if __name__ == "__main__":

    experiment = DPExperiment(
        epsilon=0.5,
        GSQ=100,
        beta=0.1,
        primary_keys=['c_id','s_id']
    )
    results = {}
    for name,query in queries.items():
        fr,t=experiment.r2t_mechanism(query)
        results[name] = (fr,t)

    for name,(fr,t) in results.items():
        relative_error = "N/A"
        if t != 0:
            relative_error = abs(fr - t) / abs(t)
        print(f"\n[Final Result of {name}]")
        print(f"Result of DP: {fr:.2f}")
        print(f"Initial Query: {t:.2f}")
        if relative_error != "N/A":
            print(f"Relative Error: {relative_error:.2%}")
        else:
            print("Initial Query is 0, Can't Calculate Relative Error")
