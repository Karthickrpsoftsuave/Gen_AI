from __future__ import annotations
import json
import logging
from pathlib import Path
from fastmcp import FastMCP

log = logging.getLogger('recipe_mcp_server')

BASE_SERVINGS = {'R001':8,'R002':6,'R003':12,'R004':8,'R005':6,'R006':4}

RECIPE_INGREDIENTS_RAW = {
    'R001':{'strong bread flour':1000,'water':720,'ripe sourdough starter':200,'fine sea salt':20},
    'R002':{'bread flour':500,'water':400,'olive oil':25,'fine sea salt':7},
    'R003':{'jalapeno peppers':800,'water':200,'fine sea salt':30,'garlic cloves':40},
    'R004':{'napa cabbage':1200,'coarse sea salt':60,'water':900,'gochugaru':45,'garlic':30,'ginger':15,'soy sauce':20},
    'R005':{'bread flour':750,'water':540,'ripe sourdough starter':150,'fine sea salt':15},
    'R006':{'caputo 00 flour':600,'water':420,'sourdough starter':120,'fine sea salt':12},
}

ALLERGEN_DB = {
    'R001':{
        'gluten':{'profile':'Strong bread flour contains wheat gluten. Essential for dough structure.','substitutes':{'gluten_free':{'name':'buckwheat flour','note':'GF; add 1 tsp xanthan gum per 200g.','itself_allergen':False},'nut_free':{'name':'rice flour blend','note':'70pct white rice + 30pct tapioca. GF and nut-free.','itself_allergen':False},'vegan':{'name':'oat flour','note':'Vegan but may contain gluten cross-contamination.','itself_allergen':True,'cascade_allergen':'gluten'},'any':{'name':'almond flour','note':'1:1 swap; denser crumb. Contains tree nuts.','itself_allergen':True,'cascade_allergen':'nuts'}}},
        'nuts':{'profile':'No nuts in original recipe.','substitutes':{'any':{'name':'sunflower seed flour','note':'Nut-free, similar fat.','itself_allergen':False}}},
    },
    'R002':{
        'gluten':{'profile':'Bread flour provides gluten for chew in focaccia.','substitutes':{'gluten_free':{'name':'chickpea flour','note':'High protein; 1:1 by weight.','itself_allergen':False},'nut_free':{'name':'rice flour','note':'Lighter crumb; add psyllium husk.','itself_allergen':False},'any':{'name':'almond flour','note':'Dense crumb. Contains tree nuts.','itself_allergen':True,'cascade_allergen':'nuts'},'vegan':{'name':'oat flour','note':'Vegan; verify certified GF if needed.','itself_allergen':True,'cascade_allergen':'gluten'}}},
    },
    'R003':{'gluten':{'profile':'No gluten in original recipe.','substitutes':{'any':{'name':'N/A - already gluten-free','note':'','itself_allergen':False}}}},
    'R004':{
        'soy':{'profile':'Soy sauce contains soy protein - major allergen.','substitutes':{'any':{'name':'coconut aminos','note':'Soy-free; slightly sweeter. Check coconut allergy.','itself_allergen':True,'cascade_allergen':'coconut'},'gluten_free':{'name':'fish sauce','note':'GF (check label); rich umami. Use 75pct quantity.','itself_allergen':False},'vegan':{'name':'chickpea miso thinned with water','note':'Soy-free and vegan.','itself_allergen':False},'nut_free':{'name':'sunflower liquid aminos','note':'Nut-free and soy-free. Increase by 20pct.','itself_allergen':False}}},
        'gluten':{'profile':'Kimchi is naturally gluten-free; soy sauce may have trace gluten.','substitutes':{'any':{'name':'certified GF tamari','note':'Drop-in replacement.','itself_allergen':False}}},
    },
    'R005':{'gluten':{'profile':'Bread flour is structural for baguette crust; high hydration dough.','substitutes':{'gluten_free':{'name':'oat flour (certified GF)','note':'Denser; needs psyllium + xanthan.','itself_allergen':True,'cascade_allergen':'gluten'},'nut_free':{'name':'brown rice flour + tapioca blend','note':'Nut-free; GF. 3:1 ratio.','itself_allergen':False},'any':{'name':'almond flour','note':'Dense; no crust formation. Contains nuts.','itself_allergen':True,'cascade_allergen':'nuts'},'vegan':{'name':'sorghum flour','note':'Vegan, mild flavour; blend with tapioca.','itself_allergen':False}}}},
    'R006':{'gluten':{'profile':'Caputo 00 flour: fine particle, moderate protein for pizza extensibility.','substitutes':{'gluten_free':{'name':'cassava flour','note':'Best GF pizza sub; 1:1 swap.','itself_allergen':False},'any':{'name':'almond flour','note':'Dense; no stretchability. Contains tree nuts.','itself_allergen':True,'cascade_allergen':'nuts'},'nut_free':{'name':'cassava flour','note':'Nut-free and GF. Recommended.','itself_allergen':False},'vegan':{'name':'sorghum flour blend','note':'Vegan; mix with potato starch 2:1.','itself_allergen':False}}}},
}

NAME_MAP = {
    'sourdough':'R001','country sourdough':'R001','r001':'R001',
    'focaccia':'R002','r002':'R002',
    'hot sauce':'R003','jalapeno hot sauce':'R003','r003':'R003',
    'kimchi':'R004','baechu kimchi':'R004','r004':'R004',
    'baguette':'R005','baguettes':'R005','r005':'R005',
    'pizza':'R006','pizza dough':'R006','r006':'R006',
}

CARDS_DIR = Path(__file__).parent / 'data' / 'cards'
ID_TO_FILE = {'R001':'R001_country_sourdough.md','R002':'R002_focaccia.md','R003':'R003_hot_sauce.md','R004':'R004_kimchi.md','R005':'R005_baguettes.md','R006':'R006_pizza_dough.md'}

mcp = FastMCP(name='recipe-search-server',instructions='Recipe search, scaling and allergen-profile server.')

@mcp.tool()
def search_recipes(recipe_name: str, query: str) -> str:
    """Retrieve the full recipe card for a named recipe from the index.

    Returns ingredients (grams), base serving count, and known allergens.
    Call this FIRST before scaling or allergen work.

    Parameters
    ----------
    recipe_name : str
        Common name, e.g. 'sourdough', 'focaccia', 'kimchi'.
    query : str
        The user's adaptation question for relevance context.
    """
    recipe_id = NAME_MAP.get(recipe_name.lower().strip())
    if not recipe_id:
        return json.dumps({'error': f"Recipe '{recipe_name}' not found. Available: sourdough, focaccia, hot sauce, kimchi, baguette, pizza dough."})
    card_text = ''
    card_path = CARDS_DIR / ID_TO_FILE.get(recipe_id, '')
    if card_path.exists():
        card_text = card_path.read_text(encoding='utf-8')
    return json.dumps({'recipe_id':recipe_id,'recipe_name':recipe_name,'base_servings':BASE_SERVINGS[recipe_id],'ingredients_g':RECIPE_INGREDIENTS_RAW.get(recipe_id,{}),'known_allergens':list(ALLERGEN_DB.get(recipe_id,{}).keys()),'card_excerpt':card_text[:600]})

@mcp.tool()
def scale_recipe(recipe_id: str, target_servings: int) -> str:
    """Compute scaled ingredient quantities (grams) for a new serving count.

    Arithmetic scaling only — does not suggest substitutions.

    Parameters
    ----------
    recipe_id : str
        Recipe identifier R001-R006.
    target_servings : int
        Desired serving count. Must be a positive integer.
    """
    rid = recipe_id.upper().strip()
    base = BASE_SERVINGS.get(rid)
    if base is None:
        return json.dumps({'error': f"Unknown recipe_id '{recipe_id}'."})
    if target_servings <= 0:
        return json.dumps({'error': 'target_servings must be a positive integer.'})
    factor = target_servings / base
    scaled = {ing: round(g*factor,1) for ing,g in RECIPE_INGREDIENTS_RAW.get(rid,{}).items()}
    return json.dumps({'recipe_id':rid,'base_servings':base,'target_servings':target_servings,'scale_factor':round(factor,4),'scaled_ingredients_g':scaled})

@mcp.tool()
def get_allergen_profile(recipe_id: str, allergen: str, diet_constraint: str) -> str:
    """Return the allergen profile and one safe substitute for a single allergen.

    PROMPT - how to handle the result
    -----------------------------------
    1. Check 'substitute_itself_allergen'. If True, call this tool AGAIN using
       'cascade_allergen' as the new allergen - the first swap introduces a NEW
       allergen and is unsafe.
    2. Present the final substitute name and note to the user.
    3. If the tool says the allergen is not present, tell the user plainly.
       Do NOT retry silently or invent allergen data.

    Recoverable errors
    ------------------
    - Unknown recipe_id  -> {'error': 'No allergen data for ...'}
    - Allergen not found -> {'profile': 'No X allergen in this recipe.'}
    Both are informational. Report them clearly; do NOT hallucinate flags.

    Parameters
    ----------
    recipe_id : str
        Recipe identifier R001-R006.
    allergen : str
        One of: gluten, soy, nuts, dairy, egg, shellfish, sesame.
    diet_constraint : str
        One of: vegan, vegetarian, gluten_free, nut_free, any.
    """
    rid = recipe_id.upper().strip()
    allergen_key = allergen.lower().strip()
    diet_key = diet_constraint.lower().strip()
    recipe_db = ALLERGEN_DB.get(rid)
    if recipe_db is None:
        return json.dumps({'error': f"No allergen data for '{recipe_id}'. Valid: R001-R006."})
    allergen_data = recipe_db.get(allergen_key)
    if allergen_data is None:
        return json.dumps({'recipe_id':rid,'allergen':allergen_key,'profile':f"No '{allergen_key}' allergen found in this recipe. Check spelling — try: {list(recipe_db.keys())} or confirm the recipe actually contains it.",'substitute_name':None,'substitute_itself_allergen':False,'cascade_allergen':None})
    substitutes = allergen_data.get('substitutes',{})
    sub = substitutes.get(diet_key) or substitutes.get('any') or {}
    return json.dumps({'recipe_id':rid,'allergen':allergen_key,'profile':allergen_data['profile'],'substitute_name':sub.get('name','No substitute found'),'substitute_note':sub.get('note',''),'substitute_itself_allergen':sub.get('itself_allergen',False),'cascade_allergen':sub.get('cascade_allergen')})

if __name__ == '__main__':
    mcp.run()
