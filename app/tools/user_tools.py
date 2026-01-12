"""
User Tools for Gemini Function Calling
=======================================

ANSWER TO QUESTION #4: Technical Implementation - Automatic Function Calling

Gemini SDK Function Calling:
- Define functions as Python callables
- Pass to GenerativeModel(tools=[...])
- Enable automatic calling: enable_automatic_function_calling=True
- Model calls function, gets result, generates response

Error Handling in Automatic Function Calling:
- If function raises exception: Model receives error message
- Model CAN retry with different parameters
- You can also return {"error": "message"} from function

Function Schema Requirements:
- Type hints required for parameters
- Docstring becomes function description
- Parameter descriptions from docstring
"""
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Store references (set by main.py on startup)
_user_store = None
_product_service = None
_db = None


def set_stores(user_store=None, product_service=None, db=None):
    """Set store references for tools to use"""
    global _user_store, _product_service, _db
    _user_store = user_store
    _product_service = product_service
    _db = db


# =============================================================================
# USER PROFILE TOOLS
# =============================================================================

def get_user_profile(user_id: str) -> dict:
    """
    Retrieve user profile including name, allergies, and preferences.

    Call this when you need to:
    - Check user's allergies before recommending products
    - Get user's name for personalized responses
    - See user's fitness goals
    - Check purchase history

    Args:
        user_id: The unique identifier for the user

    Returns:
        dict with keys: name, allergies, goals, preferences, stats
    """
    import asyncio

    async def _get():
        if _user_store is None:
            return {
                "error": None,
                "name": None,
                "allergies": [],
                "goals": [],
                "preferences": {},
                "stats": {"total_messages": 0}
            }

        user = await _user_store.get_user(user_id)
        if not user:
            return {
                "error": None,
                "name": None,
                "allergies": [],
                "goals": [],
                "preferences": {},
                "stats": {"total_messages": 0}
            }

        return {
            "error": None,
            "name": user.get("profile", {}).get("name"),
            "allergies": user.get("profile", {}).get("allergies", []),
            "goals": user.get("profile", {}).get("goals", []),
            "preferences": user.get("profile", {}).get("preferences", {}),
            "stats": user.get("stats", {})
        }

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, need to handle differently
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _get())
                return future.result()
        else:
            return asyncio.run(_get())
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return {"error": str(e)}


def update_user_profile(
    user_id: str,
    name: Optional[str] = None,
    allergies: Optional[List[str]] = None,
    goals: Optional[List[str]] = None,
    fitness_level: Optional[str] = None
) -> dict:
    """
    Update user profile information.

    Call this when user provides:
    - Their name ("მე ვარ გიორგი")
    - Allergy information ("ლაქტოზის აუტანლობა მაქვს")
    - Fitness goals ("მასის მომატება მინდა")
    - Experience level ("დამწყები ვარ")

    Args:
        user_id: The unique identifier for the user
        name: User's name (optional)
        allergies: List of allergies like ["lactose", "gluten"] (optional)
        goals: List of goals like ["muscle_gain", "weight_loss"] (optional)
        fitness_level: One of "beginner", "intermediate", "advanced" (optional)

    Returns:
        dict with success status and updated profile
    """
    import asyncio

    async def _update():
        if _user_store is None:
            return {"success": False, "error": "Database not connected"}

        profile_updates = {}
        if name is not None:
            profile_updates["name"] = name
        if allergies is not None:
            # Convert Gemini protobuf list to native Python list
            profile_updates["allergies"] = list(allergies) if hasattr(allergies, '__iter__') else [allergies]
        if goals is not None:
            # Convert Gemini protobuf list to native Python list
            profile_updates["goals"] = list(goals) if hasattr(goals, '__iter__') else [goals]
        if fitness_level is not None:
            profile_updates["fitness_level"] = fitness_level

        if not profile_updates:
            return {"success": False, "error": "No updates provided"}

        result = await _user_store.create_or_update_user(
            user_id=user_id,
            profile_updates=profile_updates
        )

        return {
            "success": True,
            "message": "პროფილი განახლდა",
            "updated_fields": list(profile_updates.keys())
        }

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _update())
                return future.result()
        else:
            return asyncio.run(_update())
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# PRODUCT SEARCH TOOLS
# =============================================================================

def search_products(
    query: str,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    in_stock_only: bool = True
) -> dict:
    """
    Search for products in the Scoop.ge catalog.

    Call this when user asks about:
    - Specific products ("გინდა პროტეინი?")
    - Categories ("რა კრეატინები გაქვთ?")
    - Price-based queries ("100 ლარამდე რა არის?")
    - Brand searches ("Optimum Nutrition")

    Args:
        query: Search query in Georgian or English
        category: Filter by category (protein, creatine, bcaa, pre_workout, vitamin, gainer)
        max_price: Maximum price in Georgian Lari
        in_stock_only: Only show in-stock products (default True)

    Returns:
        dict with products list and count
    """
    import asyncio

    async def _search():
        if _db is None:
            # Return mock data
            return {
                "products": [
                    {
                        "id": "prod_001",
                        "name": "Gold Standard Whey",
                        "price": 159.99,
                        "brand": "Optimum Nutrition",
                        "in_stock": True
                    }
                ],
                "count": 1,
                "query": query
            }

        # Translation map for Georgian queries
        query_map = {
            "პროტეინ": "protein",
            "კრეატინ": "creatine",
            "ვიტამინ": "vitamin",
            "bcaa": "bcaa",
            "პრევორკაუთ": "pre-workout",
            "გეინერ": "gainer|mass",
        }

        # Translate query
        search_term = query.lower()
        for geo, eng in query_map.items():
            if geo in search_term:
                search_term = eng
                break

        # Build MongoDB query - SECURITY: escape regex special chars
        safe_term = re.escape(search_term)
        safe_query = re.escape(query)
        mongo_query = {
            "$or": [
                {"name": {"$regex": safe_term, "$options": "i"}},
                {"name_ka": {"$regex": safe_query, "$options": "i"}},
                {"brand": {"$regex": safe_term, "$options": "i"}},
                {"category": {"$regex": safe_term, "$options": "i"}},
            ]
        }

        if category:
            mongo_query["category"] = category

        if max_price:
            mongo_query["price"] = {"$lte": max_price}

        if in_stock_only:
            mongo_query["in_stock"] = True

        cursor = _db.products.find(mongo_query).limit(10)
        products = await cursor.to_list(length=10)

        # Format results
        results = []
        for p in products:
            results.append({
                "id": p.get("id"),
                "name": p.get("name_ka", p.get("name")),
                "brand": p.get("brand"),
                "price": p.get("price"),
                "servings": p.get("servings"),
                "in_stock": p.get("in_stock"),
                "url": p.get("product_url")
            })

        return {
            "products": results,
            "count": len(results),
            "query": query
        }

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _search())
                return future.result()
        else:
            return asyncio.run(_search())
    except Exception as e:
        logger.error(f"Error searching products: {e}")
        return {"error": str(e), "products": [], "count": 0}


def get_product_details(product_id: str) -> dict:
    """
    Get detailed information about a specific product.

    Call this when user wants more details about a product
    they've seen or been recommended.

    Args:
        product_id: The product ID (e.g., "prod_001")

    Returns:
        dict with full product details including description
    """
    import asyncio

    async def _get():
        if _db is None:
            return {
                "error": "Database not connected",
                "product": None
            }

        product = await _db.products.find_one({"id": product_id})

        if not product:
            return {
                "error": f"პროდუქტი '{product_id}' ვერ მოიძებნა",
                "product": None
            }

        return {
            "error": None,
            "product": {
                "id": product.get("id"),
                "name": product.get("name"),
                "name_ka": product.get("name_ka"),
                "brand": product.get("brand"),
                "category": product.get("category"),
                "price": product.get("price"),
                "servings": product.get("servings"),
                "in_stock": product.get("in_stock"),
                "description": product.get("description"),
                "url": product.get("product_url")
            }
        }

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _get())
                return future.result()
        else:
            return asyncio.run(_get())
    except Exception as e:
        logger.error(f"Error getting product details: {e}")
        return {"error": str(e), "product": None}


# =============================================================================
# TOOL LIST FOR GEMINI
# =============================================================================

# List of all tools to pass to Gemini
GEMINI_TOOLS = [
    get_user_profile,
    update_user_profile,
    search_products,
    get_product_details,
]


# =============================================================================
# ASYNC VERSIONS (For use with Gemini's async API)
# =============================================================================

async def async_get_user_profile(user_id: str) -> dict:
    """Async version of get_user_profile"""
    if _user_store is None:
        return {
            "error": None,
            "name": None,
            "allergies": [],
            "goals": [],
            "preferences": {},
            "stats": {"total_messages": 0}
        }

    user = await _user_store.get_user(user_id)
    if not user:
        return {
            "error": None,
            "name": None,
            "allergies": [],
            "goals": [],
            "preferences": {},
            "stats": {"total_messages": 0}
        }

    return {
        "error": None,
        "name": user.get("profile", {}).get("name"),
        "allergies": user.get("profile", {}).get("allergies", []),
        "goals": user.get("profile", {}).get("goals", []),
        "preferences": user.get("profile", {}).get("preferences", {}),
        "stats": user.get("stats", {})
    }


async def async_search_products(
    query: str,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    in_stock_only: bool = True
) -> dict:
    """Async version of search_products"""
    if _db is None:
        return {"products": [], "count": 0, "query": query}

    query_map = {
        "პროტეინ": "protein",
        "კრეატინ": "creatine",
        "ვიტამინ": "vitamin",
    }

    search_term = query.lower()
    for geo, eng in query_map.items():
        if geo in search_term:
            search_term = eng
            break

    # SECURITY: escape regex special chars
    safe_term = re.escape(search_term)
    safe_query = re.escape(query)
    mongo_query = {
        "$or": [
            {"name": {"$regex": safe_term, "$options": "i"}},
            {"name_ka": {"$regex": safe_query, "$options": "i"}},
            {"brand": {"$regex": safe_term, "$options": "i"}},
            {"category": {"$regex": safe_term, "$options": "i"}},
        ]
    }

    if in_stock_only:
        mongo_query["in_stock"] = True
    if max_price:
        mongo_query["price"] = {"$lte": max_price}
    if category:
        mongo_query["category"] = category

    cursor = _db.products.find(mongo_query).limit(10)
    products = await cursor.to_list(length=10)

    results = [{
        "id": p.get("id"),
        "name": p.get("name_ka", p.get("name")),
        "brand": p.get("brand"),
        "price": p.get("price"),
        "in_stock": p.get("in_stock"),
        "url": p.get("product_url")
    } for p in products]

    return {"products": results, "count": len(results), "query": query}
