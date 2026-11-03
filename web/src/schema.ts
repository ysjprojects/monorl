import { createContext, useContext } from "react";
import type { JsonSchema, JsonSchemaProp } from "./types";

export const SchemaContext = createContext<JsonSchema | null>(null);

/** Resolve a dotted RunConfig path (e.g. "trainer.learning_rate") to its JSON-schema property. */
export function schemaProp(schema: JsonSchema | null, path: string): JsonSchemaProp | undefined {
  if (!schema) return undefined;
  let node: JsonSchemaProp | undefined = schema;
  for (const seg of path.split(".")) {
    if (!node) return undefined;
    if (node.$ref) node = schema.$defs?.[node.$ref.replace("#/$defs/", "")];
    if (node?.items) node = node.items;
    if (node?.$ref) node = schema.$defs?.[node.$ref.replace("#/$defs/", "")];
    node = node?.properties?.[seg];
  }
  if (node?.$ref) node = schema.$defs?.[node.$ref.replace("#/$defs/", "")];
  return node;
}

/** Hook: description text for a config path, with a local fallback while the schema loads. */
export function useDescription(path: string, fallback = ""): string {
  const schema = useContext(SchemaContext);
  return schemaProp(schema, path)?.description ?? fallback;
}

// ---- immutable path helpers ------------------------------------------------------------------

export function getPath(obj: unknown, path: string): unknown {
  let cur: unknown = obj;
  for (const seg of path.split(".")) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = (cur as Record<string, unknown>)[seg];
  }
  return cur;
}

export function setPath<T>(obj: T, path: string, value: unknown): T {
  const segs = path.split(".");
  const walk = (node: unknown, i: number): unknown => {
    if (i === segs.length) return value;
    const key = segs[i];
    if (Array.isArray(node)) {
      const copy = [...node];
      copy[Number(key)] = walk(node[Number(key)], i + 1);
      return copy;
    }
    const base = node && typeof node === "object" ? (node as Record<string, unknown>) : {};
    return { ...base, [key]: walk(base[key], i + 1) };
  };
  return walk(obj, 0) as T;
}
