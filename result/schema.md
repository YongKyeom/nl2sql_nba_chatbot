# SQLite Schema: nba.sqlite
## common_player_info (rows=3632)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | person_id | TEXT | 0 | None | 0 |
| 1 | first_name | TEXT | 0 | None | 0 |
| 2 | last_name | TEXT | 0 | None | 0 |
| 3 | display_first_last | TEXT | 0 | None | 0 |
| 4 | display_last_comma_first | TEXT | 0 | None | 0 |
| 5 | display_fi_last | TEXT | 0 | None | 0 |
| 6 | player_slug | TEXT | 0 | None | 0 |
| 7 | birthdate | TIMESTAMP | 0 | None | 0 |
| 8 | school | TEXT | 0 | None | 0 |
| 9 | country | TEXT | 0 | None | 0 |
| 10 | last_affiliation | TEXT | 0 | None | 0 |
| 11 | height | TEXT | 0 | None | 0 |
| 12 | weight | TEXT | 0 | None | 0 |
| 13 | season_exp | REAL | 0 | None | 0 |
| 14 | jersey | TEXT | 0 | None | 0 |
| 15 | position | TEXT | 0 | None | 0 |
| 16 | rosterstatus | TEXT | 0 | None | 0 |
| 17 | games_played_current_season_flag | TEXT | 0 | None | 0 |
| 18 | team_id | INTEGER | 0 | None | 0 |
| 19 | team_name | TEXT | 0 | None | 0 |
| 20 | team_abbreviation | TEXT | 0 | None | 0 |
| 21 | team_code | TEXT | 0 | None | 0 |
| 22 | team_city | TEXT | 0 | None | 0 |
| 23 | playercode | TEXT | 0 | None | 0 |
| 24 | from_year | REAL | 0 | None | 0 |
| 25 | to_year | REAL | 0 | None | 0 |
| 26 | dleague_flag | TEXT | 0 | None | 0 |
| 27 | nba_flag | TEXT | 0 | None | 0 |
| 28 | games_played_flag | TEXT | 0 | None | 0 |
| 29 | draft_year | TEXT | 0 | None | 0 |
| 30 | draft_round | TEXT | 0 | None | 0 |
| 31 | draft_number | TEXT | 0 | None | 0 |
| 32 | greatest_75_flag | TEXT | 0 | None | 0 |

## draft_combine_stats (rows=1633)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | season | TEXT | 0 | None | 0 |
| 1 | player_id | TEXT | 0 | None | 0 |
| 2 | first_name | TEXT | 0 | None | 0 |
| 3 | last_name | TEXT | 0 | None | 0 |
| 4 | player_name | TEXT | 0 | None | 0 |
| 5 | position | TEXT | 0 | None | 0 |
| 6 | height_wo_shoes | REAL | 0 | None | 0 |
| 7 | height_wo_shoes_ft_in | TEXT | 0 | None | 0 |
| 8 | height_w_shoes | REAL | 0 | None | 0 |
| 9 | height_w_shoes_ft_in | TEXT | 0 | None | 0 |
| 10 | weight | TEXT | 0 | None | 0 |
| 11 | wingspan | REAL | 0 | None | 0 |
| 12 | wingspan_ft_in | TEXT | 0 | None | 0 |
| 13 | standing_reach | REAL | 0 | None | 0 |
| 14 | standing_reach_ft_in | TEXT | 0 | None | 0 |
| 15 | body_fat_pct | TEXT | 0 | None | 0 |
| 16 | hand_length | TEXT | 0 | None | 0 |
| 17 | hand_width | TEXT | 0 | None | 0 |
| 18 | standing_vertical_leap | REAL | 0 | None | 0 |
| 19 | max_vertical_leap | REAL | 0 | None | 0 |
| 20 | lane_agility_time | REAL | 0 | None | 0 |
| 21 | modified_lane_agility_time | REAL | 0 | None | 0 |
| 22 | three_quarter_sprint | REAL | 0 | None | 0 |
| 23 | bench_press | REAL | 0 | None | 0 |
| 24 | spot_fifteen_corner_left | TEXT | 0 | None | 0 |
| 25 | spot_fifteen_break_left | TEXT | 0 | None | 0 |
| 26 | spot_fifteen_top_key | TEXT | 0 | None | 0 |
| 27 | spot_fifteen_break_right | TEXT | 0 | None | 0 |
| 28 | spot_fifteen_corner_right | TEXT | 0 | None | 0 |
| 29 | spot_college_corner_left | TEXT | 0 | None | 0 |
| 30 | spot_college_break_left | TEXT | 0 | None | 0 |
| 31 | spot_college_top_key | TEXT | 0 | None | 0 |
| 32 | spot_college_break_right | TEXT | 0 | None | 0 |
| 33 | spot_college_corner_right | TEXT | 0 | None | 0 |
| 34 | spot_nba_corner_left | TEXT | 0 | None | 0 |
| 35 | spot_nba_break_left | TEXT | 0 | None | 0 |
| 36 | spot_nba_top_key | TEXT | 0 | None | 0 |
| 37 | spot_nba_break_right | TEXT | 0 | None | 0 |
| 38 | spot_nba_corner_right | TEXT | 0 | None | 0 |
| 39 | off_drib_fifteen_break_left | TEXT | 0 | None | 0 |
| 40 | off_drib_fifteen_top_key | TEXT | 0 | None | 0 |
| 41 | off_drib_fifteen_break_right | TEXT | 0 | None | 0 |
| 42 | off_drib_college_break_left | TEXT | 0 | None | 0 |
| 43 | off_drib_college_top_key | TEXT | 0 | None | 0 |
| 44 | off_drib_college_break_right | TEXT | 0 | None | 0 |
| 45 | on_move_fifteen | TEXT | 0 | None | 0 |
| 46 | on_move_college | TEXT | 0 | None | 0 |

## draft_history (rows=8257)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | person_id | TEXT | 0 | None | 0 |
| 1 | player_name | TEXT | 0 | None | 0 |
| 2 | season | TEXT | 0 | None | 0 |
| 3 | round_number | INTEGER | 0 | None | 0 |
| 4 | round_pick | INTEGER | 0 | None | 0 |
| 5 | overall_pick | INTEGER | 0 | None | 0 |
| 6 | draft_type | TEXT | 0 | None | 0 |
| 7 | team_id | TEXT | 0 | None | 0 |
| 8 | team_city | TEXT | 0 | None | 0 |
| 9 | team_name | TEXT | 0 | None | 0 |
| 10 | team_abbreviation | TEXT | 0 | None | 0 |
| 11 | organization | TEXT | 0 | None | 0 |
| 12 | organization_type | TEXT | 0 | None | 0 |
| 13 | player_profile_flag | TEXT | 0 | None | 0 |

## game (rows=65698)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | season_id | TEXT | 0 | None | 0 |
| 1 | team_id_home | TEXT | 0 | None | 0 |
| 2 | team_abbreviation_home | TEXT | 0 | None | 0 |
| 3 | team_name_home | TEXT | 0 | None | 0 |
| 4 | game_id | TEXT | 0 | None | 0 |
| 5 | game_date | TIMESTAMP | 0 | None | 0 |
| 6 | matchup_home | TEXT | 0 | None | 0 |
| 7 | wl_home | TEXT | 0 | None | 0 |
| 8 | min | INTEGER | 0 | None | 0 |
| 9 | fgm_home | REAL | 0 | None | 0 |
| 10 | fga_home | REAL | 0 | None | 0 |
| 11 | fg_pct_home | REAL | 0 | None | 0 |
| 12 | fg3m_home | REAL | 0 | None | 0 |
| 13 | fg3a_home | REAL | 0 | None | 0 |
| 14 | fg3_pct_home | REAL | 0 | None | 0 |
| 15 | ftm_home | REAL | 0 | None | 0 |
| 16 | fta_home | REAL | 0 | None | 0 |
| 17 | ft_pct_home | REAL | 0 | None | 0 |
| 18 | oreb_home | REAL | 0 | None | 0 |
| 19 | dreb_home | REAL | 0 | None | 0 |
| 20 | reb_home | REAL | 0 | None | 0 |
| 21 | ast_home | REAL | 0 | None | 0 |
| 22 | stl_home | REAL | 0 | None | 0 |
| 23 | blk_home | REAL | 0 | None | 0 |
| 24 | tov_home | REAL | 0 | None | 0 |
| 25 | pf_home | REAL | 0 | None | 0 |
| 26 | pts_home | REAL | 0 | None | 0 |
| 27 | plus_minus_home | INTEGER | 0 | None | 0 |
| 28 | video_available_home | INTEGER | 0 | None | 0 |
| 29 | team_id_away | TEXT | 0 | None | 0 |
| 30 | team_abbreviation_away | TEXT | 0 | None | 0 |
| 31 | team_name_away | TEXT | 0 | None | 0 |
| 32 | matchup_away | TEXT | 0 | None | 0 |
| 33 | wl_away | TEXT | 0 | None | 0 |
| 34 | fgm_away | REAL | 0 | None | 0 |
| 35 | fga_away | REAL | 0 | None | 0 |
| 36 | fg_pct_away | REAL | 0 | None | 0 |
| 37 | fg3m_away | REAL | 0 | None | 0 |
| 38 | fg3a_away | REAL | 0 | None | 0 |
| 39 | fg3_pct_away | REAL | 0 | None | 0 |
| 40 | ftm_away | REAL | 0 | None | 0 |
| 41 | fta_away | REAL | 0 | None | 0 |
| 42 | ft_pct_away | REAL | 0 | None | 0 |
| 43 | oreb_away | REAL | 0 | None | 0 |
| 44 | dreb_away | REAL | 0 | None | 0 |
| 45 | reb_away | REAL | 0 | None | 0 |
| 46 | ast_away | REAL | 0 | None | 0 |
| 47 | stl_away | REAL | 0 | None | 0 |
| 48 | blk_away | REAL | 0 | None | 0 |
| 49 | tov_away | REAL | 0 | None | 0 |
| 50 | pf_away | REAL | 0 | None | 0 |
| 51 | pts_away | REAL | 0 | None | 0 |
| 52 | plus_minus_away | INTEGER | 0 | None | 0 |
| 53 | video_available_away | INTEGER | 0 | None | 0 |
| 54 | season_type | TEXT | 0 | None | 0 |

## game_info (rows=58053)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | game_id | TEXT | 0 | None | 0 |
| 1 | game_date | TIMESTAMP | 0 | None | 0 |
| 2 | attendance | INTEGER | 0 | None | 0 |
| 3 | game_time | TEXT | 0 | None | 0 |

## game_summary (rows=58110)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | game_date_est | TIMESTAMP | 0 | None | 0 |
| 1 | game_sequence | INTEGER | 0 | None | 0 |
| 2 | game_id | TEXT | 0 | None | 0 |
| 3 | game_status_id | INTEGER | 0 | None | 0 |
| 4 | game_status_text | TEXT | 0 | None | 0 |
| 5 | gamecode | TEXT | 0 | None | 0 |
| 6 | home_team_id | TEXT | 0 | None | 0 |
| 7 | visitor_team_id | TEXT | 0 | None | 0 |
| 8 | season | TEXT | 0 | None | 0 |
| 9 | live_period | INTEGER | 0 | None | 0 |
| 10 | live_pc_time | TEXT | 0 | None | 0 |
| 11 | natl_tv_broadcaster_abbreviation | TEXT | 0 | None | 0 |
| 12 | live_period_time_bcast | TEXT | 0 | None | 0 |
| 13 | wh_status | INTEGER | 0 | None | 0 |

## inactive_players (rows=110191)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | game_id | TEXT | 0 | None | 0 |
| 1 | player_id | TEXT | 0 | None | 0 |
| 2 | first_name | TEXT | 0 | None | 0 |
| 3 | last_name | TEXT | 0 | None | 0 |
| 4 | jersey_num | TEXT | 0 | None | 0 |
| 5 | team_id | TEXT | 0 | None | 0 |
| 6 | team_city | TEXT | 0 | None | 0 |
| 7 | team_name | TEXT | 0 | None | 0 |
| 8 | team_abbreviation | TEXT | 0 | None | 0 |

## line_score (rows=58053)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | game_date_est | TIMESTAMP | 0 | None | 0 |
| 1 | game_sequence | INTEGER | 0 | None | 0 |
| 2 | game_id | TEXT | 0 | None | 0 |
| 3 | team_id_home | TEXT | 0 | None | 0 |
| 4 | team_abbreviation_home | TEXT | 0 | None | 0 |
| 5 | team_city_name_home | TEXT | 0 | None | 0 |
| 6 | team_nickname_home | TEXT | 0 | None | 0 |
| 7 | team_wins_losses_home | TEXT | 0 | None | 0 |
| 8 | pts_qtr1_home | TEXT | 0 | None | 0 |
| 9 | pts_qtr2_home | TEXT | 0 | None | 0 |
| 10 | pts_qtr3_home | TEXT | 0 | None | 0 |
| 11 | pts_qtr4_home | TEXT | 0 | None | 0 |
| 12 | pts_ot1_home | INTEGER | 0 | None | 0 |
| 13 | pts_ot2_home | INTEGER | 0 | None | 0 |
| 14 | pts_ot3_home | INTEGER | 0 | None | 0 |
| 15 | pts_ot4_home | INTEGER | 0 | None | 0 |
| 16 | pts_ot5_home | INTEGER | 0 | None | 0 |
| 17 | pts_ot6_home | INTEGER | 0 | None | 0 |
| 18 | pts_ot7_home | INTEGER | 0 | None | 0 |
| 19 | pts_ot8_home | INTEGER | 0 | None | 0 |
| 20 | pts_ot9_home | INTEGER | 0 | None | 0 |
| 21 | pts_ot10_home | INTEGER | 0 | None | 0 |
| 22 | pts_home | REAL | 0 | None | 0 |
| 23 | team_id_away | TEXT | 0 | None | 0 |
| 24 | team_abbreviation_away | TEXT | 0 | None | 0 |
| 25 | team_city_name_away | TEXT | 0 | None | 0 |
| 26 | team_nickname_away | TEXT | 0 | None | 0 |
| 27 | team_wins_losses_away | TEXT | 0 | None | 0 |
| 28 | pts_qtr1_away | INTEGER | 0 | None | 0 |
| 29 | pts_qtr2_away | TEXT | 0 | None | 0 |
| 30 | pts_qtr3_away | TEXT | 0 | None | 0 |
| 31 | pts_qtr4_away | INTEGER | 0 | None | 0 |
| 32 | pts_ot1_away | INTEGER | 0 | None | 0 |
| 33 | pts_ot2_away | INTEGER | 0 | None | 0 |
| 34 | pts_ot3_away | INTEGER | 0 | None | 0 |
| 35 | pts_ot4_away | INTEGER | 0 | None | 0 |
| 36 | pts_ot5_away | INTEGER | 0 | None | 0 |
| 37 | pts_ot6_away | INTEGER | 0 | None | 0 |
| 38 | pts_ot7_away | INTEGER | 0 | None | 0 |
| 39 | pts_ot8_away | INTEGER | 0 | None | 0 |
| 40 | pts_ot9_away | INTEGER | 0 | None | 0 |
| 41 | pts_ot10_away | INTEGER | 0 | None | 0 |
| 42 | pts_away | REAL | 0 | None | 0 |

## officials (rows=70971)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | game_id | TEXT | 0 | None | 0 |
| 1 | official_id | TEXT | 0 | None | 0 |
| 2 | first_name | TEXT | 0 | None | 0 |
| 3 | last_name | TEXT | 0 | None | 0 |
| 4 | jersey_num | TEXT | 0 | None | 0 |

## other_stats (rows=28271)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | game_id | TEXT | 0 | None | 0 |
| 1 | league_id | TEXT | 0 | None | 0 |
| 2 | team_id_home | TEXT | 0 | None | 0 |
| 3 | team_abbreviation_home | TEXT | 0 | None | 0 |
| 4 | team_city_home | TEXT | 0 | None | 0 |
| 5 | pts_paint_home | INTEGER | 0 | None | 0 |
| 6 | pts_2nd_chance_home | INTEGER | 0 | None | 0 |
| 7 | pts_fb_home | INTEGER | 0 | None | 0 |
| 8 | largest_lead_home | INTEGER | 0 | None | 0 |
| 9 | lead_changes | INTEGER | 0 | None | 0 |
| 10 | times_tied | INTEGER | 0 | None | 0 |
| 11 | team_turnovers_home | INTEGER | 0 | None | 0 |
| 12 | total_turnovers_home | INTEGER | 0 | None | 0 |
| 13 | team_rebounds_home | INTEGER | 0 | None | 0 |
| 14 | pts_off_to_home | INTEGER | 0 | None | 0 |
| 15 | team_id_away | TEXT | 0 | None | 0 |
| 16 | team_abbreviation_away | TEXT | 0 | None | 0 |
| 17 | team_city_away | TEXT | 0 | None | 0 |
| 18 | pts_paint_away | INTEGER | 0 | None | 0 |
| 19 | pts_2nd_chance_away | INTEGER | 0 | None | 0 |
| 20 | pts_fb_away | INTEGER | 0 | None | 0 |
| 21 | largest_lead_away | INTEGER | 0 | None | 0 |
| 22 | team_turnovers_away | INTEGER | 0 | None | 0 |
| 23 | total_turnovers_away | INTEGER | 0 | None | 0 |
| 24 | team_rebounds_away | INTEGER | 0 | None | 0 |
| 25 | pts_off_to_away | INTEGER | 0 | None | 0 |

## play_by_play (rows=13592899)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | game_id | TEXT | 0 | None | 0 |
| 1 | eventnum | INTEGER | 0 | None | 0 |
| 2 | eventmsgtype | INTEGER | 0 | None | 0 |
| 3 | eventmsgactiontype | INTEGER | 0 | None | 0 |
| 4 | period | INTEGER | 0 | None | 0 |
| 5 | wctimestring | TEXT | 0 | None | 0 |
| 6 | pctimestring | TEXT | 0 | None | 0 |
| 7 | homedescription | TEXT | 0 | None | 0 |
| 8 | neutraldescription | TEXT | 0 | None | 0 |
| 9 | visitordescription | TEXT | 0 | None | 0 |
| 10 | score | TEXT | 0 | None | 0 |
| 11 | scoremargin | TEXT | 0 | None | 0 |
| 12 | person1type | REAL | 0 | None | 0 |
| 13 | player1_id | TEXT | 0 | None | 0 |
| 14 | player1_name | TEXT | 0 | None | 0 |
| 15 | player1_team_id | TEXT | 0 | None | 0 |
| 16 | player1_team_city | TEXT | 0 | None | 0 |
| 17 | player1_team_nickname | TEXT | 0 | None | 0 |
| 18 | player1_team_abbreviation | TEXT | 0 | None | 0 |
| 19 | person2type | REAL | 0 | None | 0 |
| 20 | player2_id | TEXT | 0 | None | 0 |
| 21 | player2_name | TEXT | 0 | None | 0 |
| 22 | player2_team_id | TEXT | 0 | None | 0 |
| 23 | player2_team_city | TEXT | 0 | None | 0 |
| 24 | player2_team_nickname | TEXT | 0 | None | 0 |
| 25 | player2_team_abbreviation | TEXT | 0 | None | 0 |
| 26 | person3type | REAL | 0 | None | 0 |
| 27 | player3_id | TEXT | 0 | None | 0 |
| 28 | player3_name | TEXT | 0 | None | 0 |
| 29 | player3_team_id | TEXT | 0 | None | 0 |
| 30 | player3_team_city | TEXT | 0 | None | 0 |
| 31 | player3_team_nickname | TEXT | 0 | None | 0 |
| 32 | player3_team_abbreviation | TEXT | 0 | None | 0 |
| 33 | video_available_flag | TEXT | 0 | None | 0 |

## player (rows=4815)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | id | TEXT | 0 | None | 0 |
| 1 | full_name | TEXT | 0 | None | 0 |
| 2 | first_name | TEXT | 0 | None | 0 |
| 3 | last_name | TEXT | 0 | None | 0 |
| 4 | is_active | INTEGER | 0 | None | 0 |

## team (rows=30)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | id | TEXT | 0 | None | 0 |
| 1 | full_name | TEXT | 0 | None | 0 |
| 2 | abbreviation | TEXT | 0 | None | 0 |
| 3 | nickname | TEXT | 0 | None | 0 |
| 4 | city | TEXT | 0 | None | 0 |
| 5 | state | TEXT | 0 | None | 0 |
| 6 | year_founded | REAL | 0 | None | 0 |

## team_details (rows=27)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | team_id | TEXT | 0 | None | 0 |
| 1 | abbreviation | TEXT | 0 | None | 0 |
| 2 | nickname | TEXT | 0 | None | 0 |
| 3 | yearfounded | REAL | 0 | None | 0 |
| 4 | city | TEXT | 0 | None | 0 |
| 5 | arena | TEXT | 0 | None | 0 |
| 6 | arenacapacity | REAL | 0 | None | 0 |
| 7 | owner | TEXT | 0 | None | 0 |
| 8 | generalmanager | TEXT | 0 | None | 0 |
| 9 | headcoach | TEXT | 0 | None | 0 |
| 10 | dleagueaffiliation | TEXT | 0 | None | 0 |
| 11 | facebook | TEXT | 0 | None | 0 |
| 12 | instagram | TEXT | 0 | None | 0 |
| 13 | twitter | TEXT | 0 | None | 0 |

## team_history (rows=50)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | team_id | TEXT | 0 | None | 0 |
| 1 | city | TEXT | 0 | None | 0 |
| 2 | nickname | TEXT | 0 | None | 0 |
| 3 | year_founded | INTEGER | 0 | None | 0 |
| 4 | year_active_till | INTEGER | 0 | None | 0 |

## team_info_common (rows=0)
| cid | name | type | notnull | dflt_value | pk |
|---:|---|---|---:|---|---:|
| 0 | team_id | TEXT | 0 | None | 0 |
| 1 | season_year | TEXT | 0 | None | 0 |
| 2 | team_city | TEXT | 0 | None | 0 |
| 3 | team_name | TEXT | 0 | None | 0 |
| 4 | team_abbreviation | TEXT | 0 | None | 0 |
| 5 | team_conference | TEXT | 0 | None | 0 |
| 6 | team_division | TEXT | 0 | None | 0 |
| 7 | team_code | TEXT | 0 | None | 0 |
| 8 | team_slug | TEXT | 0 | None | 0 |
| 9 | w | INTEGER | 0 | None | 0 |
| 10 | l | INTEGER | 0 | None | 0 |
| 11 | pct | REAL | 0 | None | 0 |
| 12 | conf_rank | INTEGER | 0 | None | 0 |
| 13 | div_rank | INTEGER | 0 | None | 0 |
| 14 | min_year | INTEGER | 0 | None | 0 |
| 15 | max_year | INTEGER | 0 | None | 0 |
| 16 | league_id | TEXT | 0 | None | 0 |
| 17 | season_id | TEXT | 0 | None | 0 |
| 18 | pts_rank | INTEGER | 0 | None | 0 |
| 19 | pts_pg | REAL | 0 | None | 0 |
| 20 | reb_rank | INTEGER | 0 | None | 0 |
| 21 | reb_pg | REAL | 0 | None | 0 |
| 22 | ast_rank | INTEGER | 0 | None | 0 |
| 23 | ast_pg | REAL | 0 | None | 0 |
| 24 | opp_pts_rank | INTEGER | 0 | None | 0 |
| 25 | opp_pts_pg | REAL | 0 | None | 0 |

