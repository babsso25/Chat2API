/**
 * Stream Tool Handler Module - Handle tool calls in streaming responses
 * Used by all provider-specific StreamHandlers
 * 
 * Strategy: Send all content immediately during streaming, only parse tool calls at the end
 */

import { parseToolCallsFromText } from './toolParser'

export interface ToolCallState {
  contentBuffer: string
  toolCallIndex: number
  hasEmittedToolCall: boolean
}

export function createToolCallState(): ToolCallState {
  return {
    contentBuffer: '',
    toolCallIndex: 0,
    hasEmittedToolCall: false
  }
}

/**
 * Process streaming content
 * Simple approach: accumulate content and send immediately, parse tool calls only at flush
 */
export function processStreamContent(
  content: string,
  state: ToolCallState,
  baseChunk: any,
  isFirstChunk: boolean,
  modelType: string = 'default'
): { chunks: any[], shouldFlush: boolean } {
  const result: any[] = []

  if (!content) {
    return { chunks: result, shouldFlush: false }
  }

  // Always accumulate content for final tool call detection
  state.contentBuffer += content

  // Send content immediately - no buffering during streaming
  if (!state.hasEmittedToolCall) {
    result.push({
      ...baseChunk,
      choices: [{
        index: 0,
        delta: {
          role: isFirstChunk ? 'assistant' : undefined,
          content: content
        },
        finish_reason: null
      }]
    })
  }

  return { chunks: result, shouldFlush: true }
}

/**
 * Flush any remaining content in the buffer at the end of stream
 * This is where we check for tool calls in the accumulated content
 */
export function flushToolCallBuffer(
  state: ToolCallState,
  baseChunk: any,
  modelType: string = 'default'
): any[] {
  const result: any[] = []

  console.log('[StreamToolHandler] flushToolCallBuffer called, buffer length:', state.contentBuffer?.length || 0)
  
  if (!state.contentBuffer) {
    return result
  }

  // Check for tool calls in accumulated content
  console.log('[StreamToolHandler] flushToolCallBuffer parsing:', state.contentBuffer.substring(0, 300))
  const { content: cleanContent, toolCalls } = parseToolCallsFromText(state.contentBuffer, modelType)
  console.log('[StreamToolHandler] flushToolCallBuffer parsed toolCalls:', toolCalls.length)

  if (toolCalls.length > 0) {
    // We found tool calls - but we already sent the raw content during streaming
    // This is a known limitation: the client will see raw content first, then tool calls
    // For a better UX, the client should handle this by replacing the content
    
    for (const tc of toolCalls) {
      tc.index = state.toolCallIndex++
      delete tc.rawText
      result.push({
        ...baseChunk,
        choices: [{
          index: 0,
          delta: { tool_calls: [tc] },
          finish_reason: null
        }]
      })
    }
    state.hasEmittedToolCall = true
  }

  state.contentBuffer = ''
  return result
}

/**
 * Check if we should block normal content output
 * Always returns false - we send content immediately
 */
export function shouldBlockOutput(state: ToolCallState): boolean {
  return false
}

/**
 * Create a base chunk structure for OpenAI-compatible responses
 */
export function createBaseChunk(id: string, model: string, created: number) {
  return {
    id,
    model,
    object: 'chat.completion.chunk',
    created
  }
}
