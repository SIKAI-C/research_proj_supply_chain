def currentReport(option, info, para, focus=None):
    if option == "initial":

        if "cur_sol" in info:
            print("------------------------------------------------------------------")
            print("------------------------- INITIALIZATION -------------------------")
            print("------------------------------------------------------------------")
            print("THE PARAMETERS - ")
            print("coefficient of DISTANCE      - {:<3.4f}".format(para["hyper_para"]["c_d"]))
            print("coefficient of DURATION      - {:<3.4f}".format(para["hyper_para"]["c_u"]))
            print("coefficient of INFEASIBLE    - {:<3.4f}".format(para["hyper_para"]["c_i"]))
            print("coefficient of DIVERSITY     - {:<3.4f}".format(para["hyper_para"]["r_d"]))
            print("coefficient of PROFIT        - {:<3.4f}".format(para["hyper_para"]["r_p"]))
            print("adjustment DELTA             - {:<3.4f}".format(para["hyper_para"]["delta"]))
            print("TABU TERNURE                 - {:<7d}".format(para["hyper_para"]["tabu_tenure"]))
            print("TABU GOOD                    - {:<7d}".format(para["hyper_para"]["tabu_good"]))
            print("TERMINAL ITERATIONS          - {:<7d}".format(para["hyper_para"]["terminal_max"]))
            if para["hyper_para"]["initial"]["type"] != "default":
                print("TIME LIMITS ORTOOLS          - {:<7d}".format(para["hyper_para"]["initial"]["ortools_time_limits"]))
            print("NODE AGGREGATION             - {}".format(para["hyper_para"]["aggregation"]))
            print("INITIAL SOLUTION     - ")
            for i in range(len(info["cur_sol"])):
                print("ROUTE-{:>2d}: {}".format(i, info["cur_sol"][i]))
            print("------------------------------------------------------------------")

        else:
            print("THE PARAMETERS - ")
            if focus == None:
                print("coefficient of DISTANCE      - {:<3.4f}".format(para["c_d"]))
                print("coefficient of DURATION      - {:<3.4f}".format(para["c_u"]))
                print("coefficient of INFEASIBLE    - {:<3.4f}".format(para["c_i"]))
                print("coefficient of DIVERSITY     - {:<3.4f}".format(para["r_d"]))
                print("coefficient of PROFIT        - {:<3.4f}".format(para["r_p"]))
                print("adjustment DELTA             - {:<3.4f}".format(para["delta"]))
                print("TABU TERNURE                 - {:<7d}".format(para["tabu_tenure"]))
                print("TABU GOOD                    - {:<7d}".format(para["tabu_good"]))
                print("TERMINAL ITERATIONS          - {:<7d}".format(para["terminal_max"]))
                print("NODE AGGREGATION             - {}".format(para["aggregation"]))
            else:
                if "c_d" in focus: print("coefficient of DISTANCE      - {:<3.4f}".format(para["c_d"]))
                if "c_u" in focus: print("coefficient of DURATION      - {:<3.4f}".format(para["c_u"]))
                if "c_i" in focus: print("coefficient of INFEASIBLE    - {:<3.4f}".format(para["c_i"]))
                if "r_d" in focus: print("coefficient of DIVERSITY     - {:<3.4f}".format(para["r_d"]))
                if "r_p" in focus: print("coefficient of PROFIT        - {:<3.4f}".format(para["r_p"]))
                if "delta" in focus: print("adjustment DELTA             - {:<3.4f}".format(para["delta"]))
                if "tabu_tenure" in focus: print("TABU TERNURE                 - {:<7d}".format(para["tabu_tenure"]))
                if "tabu_good" in focus: print("TABU GOOD                    - {:<7d}".format(para["tabu_good"]))
                if "terminal_max" in focus: print("TERMINAL ITERATIONS          - {:<7d}".format(para["terminal_max"]))
                if "aggregation" in focus: print("NODE AGGREGATION             - {}".format(para["aggregation"]))

    elif option == "iteration":
        print("--------------------------- ITERATION ----------------------------")
        print("CURRENT ITERATION      - {:>7d}".format(info["cur_iteration"]))
        print("EVALUTION FUNCTION     - {:>4.4f}".format(info["cur_f"]))
        print("CURRENT DISTANCE   - {:>.6e} | para - {:>.6e} | cost - {:>.6e}".\
            format(info["cur_f_info"]["tot_distance"], para["hyper_para"]["c_d"], info["cur_f_info"]["cost_of_distance"]))
        print("CURRENT DURATION   - {:>.6e} | para - {:>.6e} | cost - {:>.6e}".\
            format(info["cur_f_info"]["tot_duration"], para["hyper_para"]["c_u"], info["cur_f_info"]["cost_of_duration"]))
        print("CURRENT VIOLATION  - {:>.6e} | para - {:>.6e} | cost - {:>.6e}".\
            format(info["cur_f_info"]["tot_infeasible"], para["hyper_para"]["c_i"], info["cur_f_info"]["cost_of_infeasible"]))
        print("CURRENT DIVERSITY  - {:>.6e} | para - {:>.6e} | cost - {:>.6e}".\
            format(info["cur_f_info"]["tot_repeated_attributes"], para["hyper_para"]["r_d"], info["cur_f_info"]["reward_of_diversity"]))
        print("CURRENT PROFIT     - {:>.6e} | para - {:>.6e} | cost - {:>.6e}".\
            format(info["cur_f_info"]["maximal_profit"], para["hyper_para"]["r_p"], info["cur_f_info"]["reward_of_profit"]))
        print("CURRENT SOLUTION       - ")
        for i in range(len(info["cur_sol"])):
            print("ROUTE-{:>2d}: {}".format(i, info["cur_sol"][i]))
    elif option == "sudden_terminal":
        print("------------------------- SUDDEN TERMINAL ------------------------")
        print("THE ITERATION SUDDENLY TERMINAL AT {:>7d} STEP".format(info["cur_iteration"]))
        print("EVALUTION FUNCTION   - {:>4.4f}".format(info["cur_f"]))
        print("FINAL TOTAL DISTANCE       - {:>4.4f}".format(info["cur_f_info"]["cost_of_distance"]))
        print("FINAL TOTAL DURATION       - {:>4.4f}".format(info["cur_f_info"]["cost_of_duration"]))
        print("FINAL TOTAL VIOLATION      - {:>4.4f}".format(info["cur_f_info"]["cost_of_infeasible"]))
        print("FINAL TOTAL DIVERSITY      - {:>4.4f}".format(info["cur_f_info"]["reward_of_diversity"]))
        print("FINAL TOTAL PROFIT         - {:>4.4f}".format(info["cur_f_info"]["reward_of_profit"]))
        print("SOLUTION             - ")
        for i in range(len(info["cur_sol"])):
            print("ROUTE-{:>2d}: {}".format(i, info["cur_sol"][i]))
    elif option == "normal_terminal":
        print("---------------------------- TERMINAL ----------------------------")
        print("THE ALGORITHM REACHS THE FINAL RESULT AFTER {:>7d} STEPS".format(info["cur_iteration"]))
        print("FINAL CURRENT ITERATION    - {:>7d}".format(info["cur_iteration"]))
        print("FINAL EVALUTION FUNCTION   - {:>4.4f}".format(info["cur_f"]))
        print("FINAL TOTAL DISTANCE       - {:>4.4f}".format(info["cur_f_info"]["cost_of_distance"]))
        print("FINAL TOTAL DURATION       - {:>4.4f}".format(info["cur_f_info"]["cost_of_duration"]))
        print("FINAL TOTAL VIOLATION      - {:>4.4f}".format(info["cur_f_info"]["cost_of_infeasible"]))
        print("FINAL TOTAL DIVERSITY      - {:>4.4f}".format(info["cur_f_info"]["reward_of_diversity"]))
        print("FINAL TOTAL PROFIT         - {:>4.4f}".format(info["cur_f_info"]["reward_of_profit"]))
        print("FINAL SOLUTION             - ")
        for i in range(len(info["cur_sol"])):
            print("ROUTE-{:>2d}: {}".format(i, info["cur_sol"][i]))
    else:
        print("ERROR!! - THE OPTION ERROR!")
    print()