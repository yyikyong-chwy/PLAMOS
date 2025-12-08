#manage this as a langgraph solution, to make it easier to manage the flow of the application

from __future__ import annotations
from typing import Optional
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage
#from uuid import uuid7
from langchain_core.runnables.graph import MermaidDrawMethod


#agents
from agents.basePlanAgent import basePlanAgent
from agents.planEvalAgent import planEvalAgent, plan_eval_router
from agents.plannerAgent import plannerAgent, planner_move_router #this is the one calling llm
from agents.containerPlanPrepAgent import containerPlanPrepAgent
from agents.planMoveExecutorAgent import planMoveExecutorAgent
from agents.planMoveCritiqueAgent import planMoveCritiqueAgent, planMoveCritiqueAgent_router
from agents.planMovePadExecutorAgent import planMovePadExecutorAgent
from agents.planMoveTrimExecutorAgent import planMoveTrimExecutorAgent

#states
from states.vendorState import vendorState
from states.containerPlanState import ContainerPlanState
import preprocessing.data_preprocessing as data_preprocessing
import data.sql_lite_store as sql_lite_store
import states.state_loader as state_loader

#data storage
import data.vendor_state_store as vendor_state_store
import data.snowflake_pull as snowflake_pull

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
    graph.add_node("planMovePadExecutorAgent", planMovePadExecutorAgent)
    graph.add_node("planMoveTrimExecutorAgent", planMoveTrimExecutorAgent)
    
    graph.add_edge(START, "basePlanAgent")
    graph.add_edge("basePlanAgent", "planEvalAgent")
    graph.add_edge("containerPlanPrepAgent", "plannerAgent")
    
    # Conditional edge: route to pad executor if action is "pad", otherwise to move executor
    graph.add_conditional_edges(
        "plannerAgent",
        planner_move_router,
        {
            "planMovePadExecutorAgent": "planMovePadExecutorAgent",
            "planMoveTrimExecutorAgent": "planMoveTrimExecutorAgent",
            "planMoveExecutorAgent": "planMoveExecutorAgent",
        },
    )
    #graph.add_edge("plannerAgent", "planMoveCritiqueAgent")

    # graph.add_conditional_edges(
    #     "planMoveCritiqueAgent",
    #     planMoveCritiqueAgent_router,
    #     {
    #         "proceed": "planMoveExecutorAgent",
    #         "revise": "plannerAgent",
    #     },
    # )

    graph.add_edge("planMoveExecutorAgent", "planEvalAgent") #evaluate the plan after the move
    graph.add_edge("planMovePadExecutorAgent", "planEvalAgent") #evaluate the plan after the move
    graph.add_edge("planMoveTrimExecutorAgent", "planEvalAgent") #evaluate the plan after the move

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

#basically a wrapper for the main function workflow
def run_workflow(LOAD_FROM_SQL_LITE: bool = False):
    
    #df_sku_data, df_CBM_Max, df_kepplerSplits, demand_by_Dest = load_data()    
    #vendor_state_list = generate_vendor_states(df_sku_data, df_CBM_Max, df_kepplerSplits, demand_by_Dest)    
    vendor_state_list = data_preprocessing.run_preprocessing(LOAD_FROM_SQL_LITE) #set to false if need to load entirely from snowflake

    #iterate through the vendor_state_list and print the vendor_Code and vendor_name
    for current_vendor_state in vendor_state_list:
        print(current_vendor_state.vendor_Code, current_vendor_state.vendor_name)

        if current_vendor_state.vendor_Code != "B3755":
            continue

        config = {"configurable": {"thread_id": "test_session"},
         "recursion_limit": 500,  }
        app = compile_app()

        # app.get_graph().draw_mermaid_png(
        #     output_file_path="graph.png",
        #     draw_method=MermaidDrawMethod.API,
        # )
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

        #vendor_state_store.save_vendor_state_blob(".", current_vendor_state)
        vendor_state_store.save_vendor_state_to_db(current_vendor_state)


if __name__ == "__main__":
    run_workflow(LOAD_FROM_SQL_LITE= True)

    
        

