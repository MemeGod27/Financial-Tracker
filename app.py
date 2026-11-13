            go.Scatter(
                x=prices.index, y=prices["Close"], name="Adjusted close",
                mode="lines", line=dict(width=3),
            ),
            secondary_y=False,
        )
        if not scored.empty:
            daily = scored.groupby("date")["score"].mean().reset_index()
            chart.add_trace(
                go.Bar(
                    x=daily["date"], y=daily["score"], name="Daily news sentiment",
                    opacity=0.45,
                ),
                secondary_y=True,
            )
        chart.update_yaxes(title_text=f"Price ({currency})", secondary_y=False, showgrid=True)
        chart.update_yaxes(title_text="Sentiment (-1 to +1)", range=[-1, 1], secondary_y=True, showgrid=False)
        chart.update_layout(height=440, margin=dict(t=15, b=10, l=10, r=10), hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(chart, use_container_width=True)
        st.caption("Sentiment is shown only on dates with returned headlines. Yahoo Finance may provide a limited news history.")
        if avg is not None:
            if avg > 0.15 and day_move < -0.5:
                st.warning("Divergence: the selected-period news tone is bullish, while the latest session fell more than 0.5%. This is a flag to investigate, not proof of mispricing.")
            elif avg < -0.15 and day_move > 0.5:
                st.warning("Divergence: the selected-period news tone is bearish, while the latest session rose more than 0.5%. This is a flag to investigate, not proof of mispricing.")
            else:
                st.info("No strong headline/price divergence detected using the current thresholds.")
    with right:
        st.markdown("**Sentiment distribution**")
        if scored.empty:
            st.info("No period-matched headlines were returned.")
        else:
            counts = scored["sentiment"].value_counts()
            pie = go.Figure(go.Pie(
                labels=["Positive", "Neutral", "Negative"],
                values=[int(counts.get(x, 0)) for x in ("positive", "neutral", "negative")],
                hole=0.62,
            ))
            pie.update_layout(height=270, margin=dict(t=10, b=10, l=5, r=5), showlegend=True)
            st.plotly_chart(pie, use_container_width=True)
            st.markdown("**Daily sentiment trend**")
            daily = scored.groupby("date")["score"].mean().reset_index()
            trend = go.Figure(go.Scatter(x=daily["date"], y=daily["score"], mode="lines+markers", name="Daily average"))
            trend.add_hline(y=0, line_dash="dot")
            trend.update_yaxes(range=[-1, 1], title="Average score")
            trend.update_layout(height=220, margin=dict(t=5, b=5, l=5, r=5), showlegend=False)
            st.plotly_chart(trend, use_container_width=True)

with tab_news:
    if scored.empty:
        st.info("No recent headlines were returned for this ticker in the selected period.")
    else:
        filt = st.radio("Filter headlines", ["All", "Positive", "Neutral", "Negative"], horizontal=True)
        view = scored if filt == "All" else scored[scored["sentiment"] == filt.lower()]
        view = view.sort_values("score", key=lambda col: col.abs(), ascending=False)
        for row in view.itertuples():
            title = html.escape(row.headline)
            link = (
                f'<a href="{html.escape(row.url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
                if isinstance(row.url, str) and row.url.startswith("https://")
                else f"<b>{title}</b>"
            )
            st.markdown(
                f'<div class="news"><span class="pill {row.sentiment}">{row.sentiment.title()} {row.score:+.2f}</span>'
                f'<span class="pill neutral">{html.escape(str(row.impact))} impact</span>'
                f'<span class="meta">{html.escape(str(row.catalyst))}</span>'
                f'<div style="margin-top:7px">{link}</div>'
                f'<div class="meta">{html.escape(str(row.source))} · {row.published:%d %b %Y, %H:%M UTC}</div></div>',
                unsafe_allow_html=True,
            )
        csv_data = scored[["published", "headline", "sentiment", "score", "catalyst", "impact", "source", "url"]].to_csv(index=False)
        st.download_button("⬇️ Download analysed headlines (CSV)", csv_data, f"{ticker}_sentiment.csv", "text/csv")

with tab_catalysts:
    if scored.empty:
        st.info("Catalyst analysis needs headlines in the selected period.")
    else:
        if method != "gemini":
            st.info("Catalyst categories and impact levels are set to basic fallback defaults because Gemini analysis is unavailable.")
        a, b = st.columns(2)
        with a:
            st.markdown("**Headline catalysts**")
            catalyst_counts = scored["catalyst"].value_counts().rename_axis("Catalyst").reset_index(name="Headlines")
            st.plotly_chart(go.Figure(go.Bar(x=catalyst_counts["Catalyst"], y=catalyst_counts["Headlines"])).update_layout(height=330, margin=dict(t=10, b=10)), use_container_width=True)
        with b:
            st.markdown("**Potential news impact mix**")
            impact_order = ["High", "Medium", "Low"]
            impact_counts = scored["impact"].value_counts()
            st.plotly_chart(go.Figure(go.Pie(labels=impact_order, values=[int(impact_counts.get(x, 0)) for x in impact_order], hole=.55)).update_layout(height=330, margin=dict(t=10, b=10)), use_container_width=True)
        st.markdown("**Highest-impact headlines**")
        ranked = scored.copy()
        ranked["rank_score"] = ranked["score"].abs() * ranked["impact"].map({"High": 3, "Medium": 2, "Low": 1}).fillna(1)
        ranked = ranked.sort_values("rank_score", ascending=False).head(10)
        display = ranked[["published", "headline", "sentiment", "score", "catalyst", "impact", "source"]].copy()
        display["published"] = display["published"].dt.strftime("%d %b %Y %H:%M UTC")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("Impact ranking is a prioritisation heuristic based on Gemini's High/Medium/Low label and absolute sentiment score. It is not a forecast of price movement.")

with tab_brief:
    if scored.empty:
        st.info("The briefing needs at least one headline in the selected period.")
    elif not api_key:
        st.info("Add a Gemini API key in the sidebar to generate the analyst briefing.")
    else:
        st.caption("On-demand summary of up to five most directionally significant headlines.")
        top = scored.reindex(scored["score"].abs().sort_values(ascending=False).index).head(5)
        state_key = f"brief_{ticker}_{period_label}"
        if st.button("✨ Generate analyst briefing", type="primary"):
            try:
                with st.spinner("Preparing briefing…"):
                    st.session_state[state_key] = write_briefing(api_key, ticker, day_move, top)
            except Exception as exc:
                st.error(classify_error(exc))
        if state_key in st.session_state:
            st.markdown('<div class="brief">', unsafe_allow_html=True)
            st.markdown(st.session_state[state_key])
            st.markdown("</div>", unsafe_allow_html=True)
            with st.expander("See the headlines used"):
                for i, row in enumerate(top.itertuples(), 1):
                    st.write(f"[{i}] ({row.sentiment}, {row.score:+.2f}) {row.headline}")

st.divider()
st.caption(
    "Important: Sentiment scores, confidence labels and impact rankings are analytical aids only. "
    "They do not measure the probability of a price move and should not be used as standalone buy/sell signals."
)
