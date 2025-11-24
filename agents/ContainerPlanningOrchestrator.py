#manage this as a langgraph solution, to make it easier to manage the flow of the application

from __future__ import annotations
from typing import Optional
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage
from uuid import uuid4
from langchain_core.runnables.graph import MermaidDrawMethod


#agents
from agents.basePlanAgent import basePlanAgent
from agents.planEvalAgent import planEvalAgent, plan_eval_router
from agents.plannerAgent import plannerAgent #this is the one calling llm
from agents.containerPlanPrepAgent import containerPlanPrepAgent
from agents.planMoveExecutorAgent import planMoveExecutorAgent
from agents.planMoveCritiqueAgent import planMoveCritiqueAgent, planMoveCritiqueAgent_router

#states
from states.vendorState import vendorState
from states.containerPlanState import ContainerPlanState
import preprocessing.data_preprocessing as data_preprocessing
import data.sql_lite_store as sql_lite_store
import states.state_loader as state_loader


def compile_app():
    graph = build_graph()
    memory = MemorySaver()        
    return graph.compile(checkpointer=memory)

def build_graph() -> StateGraph:
    

    graph = StateGraph(vendorState)
    graph.add_node("basePlanAgent", basePlanAgent)
    graph.add_node("planEvalAgent", planEvalAgent)
    graph.add_node("containerPlanPrepAgent", containerPlanPrepAgent)
    graph.add_node("plannerAgent", plannerAgent)
    graph.add_node("planMoveCritiqueAgent", planMoveCritiqueAgent)
    graph.add_node("planMoveExecutorAgent", planMoveExecutorAgent)
    
    graph.add_edge(START, "basePlanAgent")
    graph.add_edge("basePlanAgent", "planEvalAgent")
    graph.add_edge("containerPlanPrepAgent", "plannerAgent")
    graph.add_edge("plannerAgent", "planMoveCritiqueAgent")

    graph.add_conditional_edges(
        "planMoveCritiqueAgent",
        planMoveCritiqueAgent_router,
        {
            "proceed": "planMoveExecutorAgent",
            "revise": "plannerAgent",
        },
    )

    graph.add_edge("planMoveExecutorAgent", "planEvalAgent") #evaluate the plan after the move

    graph.add_conditional_edges(
        "planEvalAgent",
        plan_eval_router,
        {
            "next_plan": "containerPlanPrepAgent",
            "next_move": "plannerAgent",
            "end": END,
        },
    )
    
    return graph


def load_data():

    #df_sku_data = process_demand_data()
    df_sku_data = sql_lite_store.load_table("df_sku_data")
    df_CBM_Max = sql_lite_store.load_table("CBM_Max")
    df_kepplerSplits = sql_lite_store.load_table("Keppler_Split_Perc")

    demand_by_Dest = data_preprocessing.split_base_demand_by_dest(df_sku_data, df_kepplerSplits)

    return df_sku_data, df_CBM_Max, df_kepplerSplits, demand_by_Dest

def generate_vendor_states(df_sku_data, df_CBM_Max, df_kepplerSplits, demand_by_Dest):
    
    sku_data_state_list = state_loader.df_to_chewy_sku_states(df_sku_data)
    container_plan_rows = state_loader.load_container_plan_rows(demand_by_Dest)
    vendor_state_list = state_loader.df_to_vendor_states(df_sku_data, df_CBM_Max, sku_data_state_list, container_plan_rows)

    return vendor_state_list

if __name__ == "__main__":
    df_sku_data, df_CBM_Max, df_kepplerSplits, demand_by_Dest = load_data()
    vendor_state_list = generate_vendor_states(df_sku_data, df_CBM_Max, df_kepplerSplits, demand_by_Dest)
    

    #iterate through the vendor_state_list and print the vendor_Code and vendor_name
    for current_vendor_state in vendor_state_list:
        print(current_vendor_state.vendor_Code, current_vendor_state.vendor_name)

        config = {"configurable": {"thread_id": "test_session"}}
        app = compile_app()

        app.get_graph().draw_mermaid_png(
            output_file_path="graph.png",
            draw_method=MermaidDrawMethod.API,
        )
        state = app.invoke(current_vendor_state, config=config)
        current_vendor_state = vendorState.model_validate(state)        
        #print("\n\nContainer Plan Rows:\n", current_vendor_state.container_plans[0].to_df())
        print("\n Scores:")
        print("Overall Score: ", current_vendor_state.container_plans[0].metrics.overall_score)
        print("Utilization Score: ", current_vendor_state.container_plans[0].metrics.avg_utilization)        
        print("Planned Score: ", current_vendor_state.container_plans[0].metrics.ape_vs_planned)
        print("Base Score: ", current_vendor_state.container_plans[0].metrics.ape_vs_base)
        print("number of containers: ", current_vendor_state.container_plans[0].metrics.containers)
        print("\n\n")

    
        

