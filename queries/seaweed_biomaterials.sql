-- Seaweed-based recipes and how many have each measured property.
-- Run: duckdb data/biomaterials.duckdb < queries/seaweed_biomaterials.sql
WITH seaweed AS (
    SELECT DISTINCT recipe_id,
        CASE
            WHEN ingredient LIKE '%alginate%'            THEN 'alginate'
            WHEN ingredient LIKE '%carrageenan%'         THEN 'carrageenan'
            WHEN regexp_matches(ingredient, '\bagar\b')  THEN 'agar'
            WHEN ingredient = 'seaweed'                  THEN 'whole seaweed'
            ELSE 'other seaweed'
        END AS base
    FROM ingredients
    WHERE regexp_matches(ingredient, 'alginate|carrageenan|\bagar\b|seaweed')
)
SELECT p.property,
       p.unit,
       count(DISTINCT p.recipe_id)          AS n_recipes,
       count(DISTINCT r.source)             AS n_sources,
       string_agg(DISTINCT s.base, ', ')    AS bases
FROM properties p
JOIN seaweed s USING (recipe_id)
JOIN recipes r USING (recipe_id)
WHERE p.value IS NOT NULL
GROUP BY ALL
ORDER BY n_recipes DESC;
